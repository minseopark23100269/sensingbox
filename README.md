# sensingbox
표정과 손 움직임을 감지해서 원하는 기능을 쓸 수 있는 영상box

이 프로젝트는 **OpenCV**와 **MediaPipe**를 기반으로 하여, 웹캠으로부터 실시간 영상을 받아 얼굴 및 손의 랜드마크를 추출한 후, 여러 가지 효과(안경, 왕관, 카툰filter, Funny Face, 표정에 따른 이모지, 확대 및 축소)를 적용하는 시스템입니다.  

## 폴더 구조

프로젝트 폴더의 구조는 다음과 같습니다:

```
sensingbox-main/
├── 1234548.py         # 메인 파이썬 스크립트
├── images/            # 오버레이 이미지 폴더
│   ├── crown.png      # 왕관 오버레이 이미지
│   ├── glasses.png    # 안경 오버레이 이미지
│   ├── smile.png      # 웃음 효과 오버레이 이미지
│   └── zzz.png        # 눈 감은 효과 오버레이 이미지
└── output/            # 실행 시 생성됨 - 처리된 영상이 저장됩니다.
```


## 주요 기능

- **실시간 얼굴 및 손 추적:**  
  MediaPipe Face Mesh와 Hands 솔루션을 사용하여 얼굴과 손의 랜드마크를 실시간으로 추출합니다.

- **오버레이 효과:**  
  - **안경 효과:** 빨간색 트리거 원을 가리키면 얼굴에 안경이 5초간 쓸 수 있습니다. 
  - **왕관 효과:** 흰색 트리거 원을 가리키면 얼굴 상단에 왕관 이미지가 5초간 쓸 수 있습니다.
  - **카툰 효과:** 노란색 트리거 원을 가리키면 OpenCV의 스타일화 필터가 적용되어 카툰 느낌을 줍니다.
  - **Funny Face 효과:** 초록색 트리거 원을 가리키면 얼굴 영역의 **눈과 입만 확대**하는 효과를 5초간 적용됩니다.
  - **Smile/Closed-eye 효과:** 얼굴 표정을 분석하여 웃을 땐 웃는 이모지, 눈을 감을 땐 조는 이모지가 나타납니다. 

- **손 제스처 기반 확대/축소:**  
  전체 손이 펼쳐지면 영상의 화면이 확대되고, 주먹을 쥐면 화면이 원래 크기로 축소됩니다.

- **영상 저장:**  
  최종 처리된 영상은 상대 경로의 `output/recorded_video.avi` 파일에 저장됩니다.

## Dependencies

- Python 3.x
- [OpenCV-Python](https://pypi.org/project/opencv-python/)
- [MediaPipe](https://pypi.org/project/mediapipe/)
- [NumPy](https://pypi.org/project/numpy/)

## 설치 및 실행 방법

1. **저장소 클론 및 폴더 구조 확인:**

   ```bash
   git clone https://github.com/minseopark23100269/sensingbox.git
   cd sensingbox-main
필요 패키지 설치:

bash
pip install opencv-python mediapipe numpy

이미지 파일 준비:

images/ 폴더 안에 아래 이미지 파일들이 있습니다:

crown.png

glasses.png

smile.png

zzz.png

스크립트 실행:

bash
python sensingbox.py

웹캠이 켜지면서 “Effects & Hand Zoom” 창이 열리고, 실시간 영상에 효과가 적용됩니다.

손 제스처로 확대/축소, 그리고 각 트리거 원(빨강, 흰, 노랑, 초록)을 통해 효과를 활성화시킬 수 있습니다.

ESC 키를 누르면 영상이 종료되며, output/recorded_video.avi 파일에 저장됩니다.

코드 동작 방식
실시간 영상 처리: 웹캠으로부터 영상을 받아, MediaPipe로 얼굴 및 손 랜드마크를 추출합니다.

손 제스처:

is_hand_open()와 is_fist() 함수를 사용하여 손의 전체 상태를 판단합니다.

손이 펼쳐졌을 때 확대, 주먹일 때 축소하도록 scale factor를 조절합니다.

효과 트리거: 화면 상단의 4개의 트리거 원(빨강, 흰, 노랑, 초록)을 보여주며, 손의 INDEX_FINGER_TIP이 해당 원 근처에 있을 경우 관련 효과의 타이머를 갱신합니다.

Funny Face 효과: 초록색 원이 활성화되면, 얼굴 영역에서 눈과 입 영역만 선택적으로 확대하는 enlarge_feature() 함수를 이용하여 Funny Face 효과를 5초간 적용합니다.

추가 오버레이: 웃음과 눈 감은 효과는 얼굴 표정을 분석하여 별도로 적용됩니다.

영상 저장 및 확대/축소 처리: 최종 프레임에 대해 손 제스처에 따른 확대/축소를 적용한 후 중앙에서 크롭하여 출력하고, 동시에 영상을 저장합니다.


