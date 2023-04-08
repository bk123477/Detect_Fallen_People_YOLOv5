# Detect_Fallen_People_YOLOv5
1. 서론
최근 선진국의 고령화로 인하여 노인 요양 시설에 대한 수요가 급증하고 있다. 요양 시설의 환자들이 침대에서 내려오다가 넘어지거나 물건에 걸려 넘어지는 낙상 사고가 발생한 경우, 관리자에게 바로 사고 발생 사실을 알려야 즉각적인 조치가 가능하다. 하지만 관리인이 24시간 환자를 관찰할 수는 없기에 사고 사실을 알리기 위한 버튼 또는 압력 매트가 사용되고 있다. 기존 방법들은 낙상 사고 발생 시, 환자 본인이나 다른 사람이 대신 버튼을 눌러야 하고, 유지 관리가 필요하다는 단점이 있다.
컴퓨터 비전을 통한 감지 시스템은 벽이나 천장에 설치하는 형식으로 한번 설치하면 사용자의 개입이나 유지 관리가 많이 필요하지 않다. 또한, 밤낮이나 조명 여부등 외부 환경에 상관없이 24시간 모니터링이 가능하며 사고 발생 시 자동으로 기존의 알림 인프라나 문자 등 다양한 형태로 알림을 전송할 수 있다.
낙상 감지를 위한 기존의 딥 러닝 기반 솔루션은 대규모 데이터셋을 이용하기 때문에 리소스적으로 불리하고, 시스템을 구축하는 데 많은 비용이 들어 상업적인 용도로 사용하기 부적합하다. 또한, 최근 프라이버시 문제도 대두되고 있는 만큼 영상 카메라의 사용도 제한적이다.
따라서 이 연구에서는 저가형 싱글보드 컴퓨터인 Rasberry Pi 3와 깊이 및 적외선 데이터를 실시간으로 수집할 수 있는 Kinect 센서를 이용하여 실시간으로 낙상을 감지하고 알림을 전송하는 시스템을 제안을 목표로 하였다.
![image](https://user-images.githubusercontent.com/59231609/230704706-5878b4aa-3dcd-433f-b233-b11815e6fb61.png)
