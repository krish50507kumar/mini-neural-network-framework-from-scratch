class Module:
    def parameters(self):
        params = []
        for attr in self.__dict__.values():
            if isinstance(attr, dict):  # params
                params.append(attr)
        return params