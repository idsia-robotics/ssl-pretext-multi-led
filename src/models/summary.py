from src.models import model_registry

if __name__ == '__main__':
    from torchinfo import summary

    for model in model_registry.values():
        model = model(task = 'pose')
        summary(model, input_size=(1, 3, 360, 640))
        print()