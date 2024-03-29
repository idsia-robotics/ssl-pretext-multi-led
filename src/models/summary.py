from src.models import model_registry

if __name__ == '__main__':
    from torchscan import summary

    for model in model_registry.values():
        model = model(task = 'pose_and_led')
        print(model.__class__.__name__)
        summary(model, (3, 360, 640), receptive_field=True)
        print()