from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(
    image_size=160,
    margin=0,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    device=device,
    keep_all=True,
)

resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


def next(item: bytes, shape: list[int], position):
    img = torch.frombuffer(item, dtype=torch.uint8).reshape(*shape)
    crop = mtcnn(img)
    if crop is None:
        return ([], position)
    return (resnet(crop).tolist(), position)
