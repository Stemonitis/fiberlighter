PROCESSORS = {}


def register_processing(name):
    def decorator(cls):
        PROCESSORS[name] = cls
        return cls

    return decorator