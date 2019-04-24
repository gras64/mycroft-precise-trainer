from mycroft import MycroftSkill, intent_file_handler


class PreciseTrainer(MycroftSkill):
    def __init__(self):
        MycroftSkill.__init__(self)

    @intent_file_handler('trainer.precise.intent')
    def handle_trainer_precise(self, message):
        self.speak_dialog('trainer.precise')


def create_skill():
    return PreciseTrainer()

