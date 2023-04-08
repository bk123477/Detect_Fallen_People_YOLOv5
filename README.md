# Detect_Fallen_People_YOLOv5
## 1. 서론

최근 선진국의 고령화로 인하여 노인 요양 시설에 대한 수요가 급증하고 있다. 요양 시설의 환자들이 침대에서 내려오다가 넘어지거나 물건에 걸려 넘어지는 낙상 사고가 발생한 경우, 관리자에게 바로 사고 발생 사실을 알려야 즉각적인 조치가 가능하다. 하지만 관리인이 24시간 환자를 관찰할 수는 없기에 사고 사실을 알리기 위한 버튼 또는 압력 매트가 사용되고 있다. 기존 방법들은 낙상 사고 발생 시, 환자 본인이나 다른 사람이 대신 버튼을 눌러야 하고, 유지 관리가 필요하다는 단점이 있다.

컴퓨터 비전을 통한 감지 시스템은 벽이나 천장에 설치하는 형식으로 한번 설치하면 사용자의 개입이나 유지 관리가 많이 필요하지 않다. 또한, 밤낮이나 조명 여부등 외부 환경에 상관없이 24시간 모니터링이 가능하며 사고 발생 시 자동으로 기존의 알림 인프라나 문자 등 다양한 형태로 알림을 전송할 수 있다.

낙상 감지를 위한 기존의 딥 러닝 기반 솔루션은 대규모 데이터셋을 이용하기 때문에 리소스적으로 불리하고, 시스템을 구축하는 데 많은 비용이 들어 상업적인 용도로 사용하기 부적합하다. 또한, 최근 프라이버시 문제도 대두되고 있는 만큼 영상 카메라의 사용도 제한적이다.

따라서 이 연구에서는 저가형 싱글보드 컴퓨터인 Rasberry Pi 3와 깊이 및 적외선 데이터를 실시간으로 수집할 수 있는 Kinect 센서를 이용하여 실시간으로 낙상을 감지하고 알림을 전송하는 시스템을 제안을 목표로 하였다.

## 2-2. Yolo 사용 이유
ShuffleNet v2 알고리즘은 경량 알고리즘 연구를 통해 탄생한 알고리즘으로, 합성곱 신경망(CNN)의 가장 큰 연산을 요구하는 합성곱 필터의 연산을 줄인 알고리즘이다. 입력 채널과 필터의 합성곱 연산을 하기 전에 입력 채널을 작게 나눈 뒤 섞은 다음에 합성곱을 진행하는 방식이다. 이를 이용하면 전체 입력 채널에 대해서 합성곱 연산을 수행 하는 것에 비해 채널의 수 만큼 연산량을 줄이는 효과를 얻게 된다. 기존 연구에서는 Rasberry Pi를 통해 학습을 진행하였는데 이는 Rasberry Pi의 성능과 연관이 있다. 개별 합성곱의 성능은 전체 합성곱 연산에 비해 높지 않지만 연산량을 줄여주는 ShuffleNet v2를 활용한 알고리즘을 채택하였다.

하지만, 실제로 ShuffleNet v2를 통해 학습을 진행해본 결과, tensorflow의 구형 버전(tensorflow 1)을 사용하기 때문에 특정 모듈이 사용 불가한 문제가 있었고, training set과 validation set을 전처리하는 과정이 별도로 존재하여 번거로웠다. 또한, 라벨링 과정에서 모든 데이터에 대해 일일히 좌표를 찍어주어야 하는 어려움, 결정적으로 training 과정에서 적어도 100회 이상의 epoch를 실행시켜야지 의미있는 model이 나오는데, dataset이 너무 무겁고, local 컴퓨터에서는 해당 training을 수행하는 데에 버거움이 있었다.

따라서, custom data에 대한 레퍼런스들이 많고, 웹 상에서 라벨링 후 해당 image들을 기반으로 바로 모델 생성이 가능한 PyTorch 프레임워크를 사용한 YOLOv5를 이용하여 본 연구를 진행하기로 결정하였다.

## 3-1. 연구 진행 상황

먼저, 효율적인 학습을 위해 모델이 처리하는 영역을 좁혀야 하기 때문에 본 연구에서는 labeling 과정에서 class domain을 다음과 같이 5개로 설정하였다.

💡 < Chair, Couch, Fallen person, Lying Person, Normal Person >

custom dataset은 여러 open research dataset들을 활용하였다. 이러한 자료들을 roboflow를 이용하여 라벨링을 직접 진행한 이후, 학습에 사용하였다.

Yolo를 통한 학습 과정은 다음과 같다. 학습 환경은 Google Colab에서 Yolov5를 마운트하여 진행하였다. Yolov5와 roboflow를 Colab상에서 설치하면 바로 Yolo 모델과 roboflow의 데이터를 직접 가져와 사용을 할 수 있다. Yolov5의 train.py, detect.py, val.py를 통해 각각 데이터 셋에 대해 학습, 객체 탐지, 평가를 진행할 수 있다. train.py를 이용하여 학습을 진행하게 되면 학습이 완료된 Yolo 모델 두 가지가 만들어진다. best.pt, last.pt 두 가지 모델이 나오게 되는데, 각 모델에는 Yolo모델의 가중치가 담겨있다. best.pt 모델은 가장 최적이라고 생각이 되는 가중치가 담겨 있고, last.pt는 마지막 Epoch 수행 시 나오게 되는 가중치가 담기게 된다. 연구 에서는 best.pt 모델을 사용하였다.

<img width="323" alt="image" src="https://user-images.githubusercontent.com/59231609/230704782-a89078ce-b3b9-4b46-929f-c9a0a0570895.png">
Yolov5 val.py를 이용한 Confusion Matrix 결과

<img width="364" alt="image" src="https://user-images.githubusercontent.com/59231609/230704814-97695c14-7487-4609-9c0b-48fa152f5b96.png">
또한, confidence 값에 따른 precision과 recall의 변화를 그래프로 표현한 PR 곡선이다. 이 그래프에서 mAP50(각 class에 대한 AP 값들의 평균)이 0.858에 수렴함을 확인할 수 있었다. 더 나아가, IoU값을 50%부터 95%까지 5%씩 증가시키면서 수행한 mAP 값들의 평균을 계산한 mAP50-95에 대해서도 0.701의 성능을 냈다. 이를 통해 충분한 데이터셋이 추가된다면 학습 모델의 정확성을 늘릴 수 있다는 것을 확인할 수 있었다.

## 4. 관련 연구 및 참고 자료
1. Effective Deep-Learning-Based Depth Data Analysis on Low-Power Hardware for Supporting Elderly Care(Christopher Pramerdorfer Cogvis)

3. https://github.com/ultralytics/yolov5

5. https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

