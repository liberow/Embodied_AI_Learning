class BaseEnvironment:
    def __init__(self, xml_path):
        self.xml_path = xml_path

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        pass