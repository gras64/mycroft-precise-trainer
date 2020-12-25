# Copyright 2019 Mycroft AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from math import sqrt
from time import sleep

import shutil
import subprocess
import wave
from functools import partial
from glob import glob
from os import makedirs
import posixpath

from os.path import expanduser, join

import platform
import requests
from petact import install_package
from subprocess import call
from tempfile import mkstemp, mkdtemp
from threading import Thread, Event

from mycroft import MycroftSkill, intent_file_handler, Message
from mycroft.util import play_audio_file, play_wav, \
    resolve_resource_file, record
from mycroft.audio import wait_while_speaking


class PreciseTrainer(MycroftSkill):
    urls = {
        'x86_64': 'https://github.com/MycroftAI/mycroft-precise/releases'
                  '/download/v0.3.0/precise-all_0.3.0_x86_64.tar.gz',
        'armv7l': 'https://github.com/MycroftAI/mycroft-precise/releases'
                  '/download/v0.3.0/precise-all_0.3.0_armv7l.tar.gz',
    }
    train_model_base_url = 'https://raw.githubusercontent.com/MycroftAI' \
                           '/precise-data/models/{wake_word}.train.tar.gz'

    # TODO: Replace with downloaded data
    noise_folder = ''
    chunk_size = 2048
    threshold = 0.1

    def __init__(self):
        MycroftSkill.__init__(self)
        self.url = self.urls.get(platform.machine(), '')
        self.platform_supported = bool(self.url)
        if self.url and not self.url.endswith('.tar.gz'):
            self.url = requests.get(self.url).text.strip()

        self.folder = expanduser('~/.mycroft/precise-trainer')
        self.precise_config = self.config_core['precise']
        self.model_url = self.train_model_base_url.format(
            wake_word='hey-mycroft'
        )
        self.model_file = join(self.folder, posixpath.basename(
            self.model_url)).replace(
            '.tar.gz', '.net'
        )

        self.exe_folder = join(self.folder, 'precise')
        self.engine_exe = join(self.exe_folder, 'precise-engine')

        makedirs(self.folder, exist_ok=True)
        self.install_thread = Thread(target=self.install_package)
        self.install_thread.start()
        self.install_complete = Event()
        self.install_failed = False

    def on_download(self, name):
        self.log.info('Download for {} started!'.format(name))
        while not self.install_complete.is_set():
            self.log.info('Still downloading {}...'.format(name))
            sleep(5)
        self.log.info('Download of {} {}!'.format(
            name, 'failed' if self.install_failed else 'completed'
        ))

    def install_package(self):
        if not self.url:
            return
        self.install_failed = True
        try:
            install_package(self.url, self.folder, on_download=lambda: Thread(
                target=partial(
                    self.on_download, 'precise training exe'
                ), daemon=True
            ).start())
            install_package(
                self.model_url, self.folder,
                on_download=lambda: Thread(
                    target=partial(
                        self.on_download, 'precise training model'
                    ), daemon=True
                ).start()
            )
            self.install_failed = False
        finally:
            self.install_complete.set()

    def handle_precise_download(self):
        if self.install_thread.is_alive():
            self.speak_dialog('download.in.progress')
            wait_while_speaking()
            self.install_complete.wait()
            self.speak_dialog('download.complete')
            return True
        return False

    def handle_train(self, subfolder, dialog_file):
        if self.handle_precise_download():
            return

        name = self.get_response(
            'ask.speaker.name', validator=lambda x: x and len(x.split()) < 4,
            on_fail=lambda utterance: self.dialog_renderer.render(
                'name.error', {'name': utterance}
            )
        )
        if not name:
            return

        self.speak_dialog(dialog_file)
        wait_while_speaking()

        from precise_runner import PreciseEngine
        engine = PreciseEngine(self.engine_exe, self.model_file,
                               self.chunk_size)
        engine.start()

        recording = self.record_wav()

        with wave.open(recording, 'r') as wr:
            orig_params = wr.getparams()
            frames = wr.readframes(wr.getnframes() - 1)

        ww_positions = self.extract_ww_positions(frames, engine)
        engine.stop()

        samples_folder = join(self.folder, 'samples', name)
        samples_raw_folder = join(samples_folder, 'not-wake-word')
        makedirs(samples_raw_folder, exist_ok=True)
        self.split_recording(frames, samples_raw_folder, ww_positions, orig_params)

        self.speak_dialog('recording.complete')
        models_folder = join(self.folder, 'user-models')
        makedirs(models_folder, exist_ok=True)
        model_file = join(models_folder, '{}.{}.net'.format('hey-mycroft',
                                                            name))
        self.transfer_train(samples_folder, model_file)
        self.speak_dialog('model.confirm')

        thresh = self.calc_thresh(model_file, samples_raw_folder)
        print("THRESH:", thresh)

    @intent_file_handler('trainer.precise.intent')
    def handle_trainer_precise(self, message):
        self.handle_train('wake-word', 'trainer.precise')

    @intent_file_handler('train.precise.inhibit.intent')
    def handle_train_precise_inhibit(self):
        self.handle_train('not-wake-word', 'record.inhibit.precise')

    @intent_file_handler('reload.custom.precise.intent')
    def handle_reload_precise(self, message):
        name = message.data['user']
        models_folder = join(self.folder, 'user-models')
        makedirs(models_folder, exist_ok=True)
        model_file = join(models_folder, '{}.{}.net'.format('hey-mycroft',
                                                            name))
        samples_folder = join(self.folder, 'samples', name)
        samples_raw_folder = join(samples_folder, 'wake-word')
        self.transfer_train(samples_folder, model_file)
        self.speak_dialog('model.confirm')

        thresh = self.calc_thresh(model_file, samples_raw_folder)
        print("THRESH:", thresh)

    def split_recording(self, frames, folder, positions, params):
        prev_pos = 0
        for pos, end_pos in positions:
            data = frames[prev_pos:end_pos]
            prev_pos = end_pos
            sample_file = join(folder, 'sample-{}.wav'.format(pos))
            with wave.open(sample_file, 'wb') as wf:
                wf.setparams(params)
                wf.writeframes(data)

    def extract_ww_positions(self, frames, engine):
        max_pos = -1
        max_val = float('-inf')
        max_positions = []
        for i in range(self.chunk_size, len(frames) + 1, self.chunk_size):
            chunk = frames[i - self.chunk_size:i]
            prob = engine.get_prediction(chunk)
            self.log.info("PROB: {}".format(prob))
            if prob > self.threshold:
                if prob > max_val:
                    max_val = prob
                    max_pos = i
            else:
                if max_pos >= 0:
                    max_positions.append((max_pos, i))
                    max_pos = -1
                    max_val = float('-inf')

        if max_pos >= 0:
            max_positions.append((max_pos, len(frames)))

        return max_positions

    def record_wav(self):
        audio_file = resolve_resource_file(
            self.config_core.get('sounds').get('start_listening'))
        if audio_file:
            play_wav(audio_file).wait()

        self.bus.emit(Message('mycroft.mic.mute'))
        try:
            fd, tmp_file = mkstemp('.wav')
            subprocess.Popen(
                ["arecord", "-f", "S16_LE", "-r", str(16000), "-c",
                 str(1), "-d",
                 str(10), tmp_file]).wait()
        finally:
            self.bus.emit(Message('mycroft.mic.unmute'))
        return tmp_file

    def calc_thresh(self, model_file, samples_folder):
        from precise_runner import PreciseEngine
        engine = PreciseEngine(self.engine_exe, model_file, self.chunk_size)
        engine.start()

        all_max = []
        for sample_file in glob(join(samples_folder, '*.wav')):
            with wave.open(sample_file, 'r') as wr:
                frames = wr.readframes(wr.getnframes() - 1)
            chop = len(frames) % self.chunk_size
            max_pred = float('-inf')
            for i in range(10): 
                engine.get_prediction(b'\0' * self.chunk_size)
            for pos in range(chop + self.chunk_size, len(frames) + 1,
                             self.chunk_size):
                pred = engine.get_prediction(frames[pos - self.chunk_size:pos])
                max_pred = max(max_pred, pred)
            print('MAX PRED:', max_pred)
            all_max.append(max_pred)
        av_max = sum(all_max) / len(all_max)
        stddev = sqrt(sum([(i - av_max) ** 2 for i in all_max]))
        good_max = [i for i in all_max if i > av_max - stddev]
        good_av = sum(good_max) / len(good_max)
        stddev = sqrt(sum([(i - good_av) ** 2 for i in good_max]))
        thresh = good_av - stddev
        return thresh

    def transfer_train(self, samples_folder, model_file):
        noised_folder = mkdtemp()
        wake_word_folder = join(noised_folder, 'wake-word')
        not_wake_word_folder = join(noised_folder, 'not-wake-word')
        makedirs(wake_word_folder, exist_ok=True)
        makedirs(not_wake_word_folder, exist_ok=True)
        call([
            join(self.exe_folder, 'precise-add-noise'),
            samples_folder, self.noise_folder, wake_word_folder,
            '-if', '10', '-nl', '0.0', '-nh', '0.4'
        ])
        call([
            join(self.exe_folder, 'precise-add-noise'),
            self.noise_folder, self.noise_folder, not_wake_word_folder,
            '-if', '10', '-nl', '0.0', '-nh', '0.4'
        ])
        shutil.copy(self.model_file, model_file)

        call([
            join(self.exe_folder, 'precise-train'),
            model_file, noised_folder,
            '-e', '1', '-b', '4096',
        ])


def create_skill():
    return PreciseTrainer()
