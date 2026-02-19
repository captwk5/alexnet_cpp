# AlexNet C++

> AlexNet 신경망을 순수 C++로 구현한 프로젝트입니다.  
> PyTorch에서 사전 학습된(pretrained) 가중치를 바이너리 파일로 변환하여 C++ 환경에서 학습(training) 및 추론(inference)을 수행합니다.

---

## 📌 개요

이 프로젝트는 딥러닝 프레임워크(PyTorch, TensorFlow 등) 없이 **순수 C++**로 AlexNet을 구현합니다.  
CNN(Convolutional Neural Network)의 핵심 연산인 **합성곱(Convolution)**, **맥스 풀링(Max Pooling)**, **완전 연결 계층(Fully Connected Layer)**, **역전파(Backpropagation)**, **가중치 업데이트(Weight Update)** 등을 직접 구현하여 딥러닝의 내부 동작을 이해하는 데 목적을 두고 있습니다.

---

## 🏗️ 네트워크 아키텍처

PyTorch의 AlexNet 구조를 기반으로 하며, 다음과 같은 레이어로 구성됩니다:

| Layer | Type | Input | Output | Kernel | Stride |
|-------|------|-------|--------|--------|--------|
| 1 | Conv2D + ReLU | 3 × 224 × 224 | 64 × 56 × 56 | 11 × 11 | 4 |
| 2 | MaxPool2D | 64 × 56 × 56 | 64 × 27 × 27 | 3 × 3 | 2 |
| 3 | Conv2D + ReLU | 64 × 27 × 27 | 192 × 27 × 27 | 5 × 5 | 1 |
| 4 | MaxPool2D | 192 × 27 × 27 | 192 × 13 × 13 | 3 × 3 | 2 |
| 5 | Conv2D + ReLU | 192 × 13 × 13 | 384 × 13 × 13 | 3 × 3 | 1 |
| 6 | Conv2D + ReLU | 384 × 13 × 13 | 256 × 13 × 13 | 3 × 3 | 1 |
| 7 | Conv2D + ReLU | 256 × 13 × 13 | 256 × 13 × 13 | 3 × 3 | 1 |
| 8 | MaxPool2D | 256 × 13 × 13 | 256 × 6 × 6 | 3 × 3 | 2 |
| 9 | Flatten | 256 × 6 × 6 | 9216 | - | - |
| 10 | FC + ReLU | 9216 | 4096 | - | - |
| 11 | FC + ReLU | 4096 | 4096 | - | - |
| 12 | FC (Output) | 4096 | 1000 | - | - |

최종 출력은 **Softmax**를 통해 1000개 클래스에 대한 확률 분포로 변환됩니다.  
손실 함수는 **Cross-Entropy Loss**를 사용합니다.

---

## 📂 프로젝트 구조

```
alexnet_cpp/
├── CMakeLists.txt            # CMake 빌드 설정
├── main.cpp                  # 프로그램 진입점
├── cnn.hpp                   # CNN 기본 클래스 헤더 (공통 연산 인터페이스)
├── cnn.cpp                   # CNN 기본 클래스 구현 (Convolution, Pooling, FC, Backprop 등)
├── utils.cpp                 # 유틸리티 함수 (메모리 관리, 가중치 로드, 문자열 파싱)
├── alexnet/
│   ├── alexnet.hpp           # AlexNet 클래스 헤더 (네트워크 구조 정의)
│   └── alexnet.cpp           # AlexNet 클래스 구현 (학습 루프, Forward/Backward Pass)
└── python/
    ├── pytorch_network.py    # PyTorch AlexNet 모델 정의
    ├── pytorch_train_test.py # PyTorch 학습 및 테스트 스크립트
    ├── get_init_weight.py    # PyTorch → 바이너리 가중치 추출
    └── set_trained_weight.py # 바이너리 가중치 → PyTorch 모델 로드 및 검증
```

### 주요 파일 설명

| 파일 | 설명 |
|------|------|
| **`cnn.hpp / cnn.cpp`** | CNN 연산의 기본 클래스. 합성곱, 풀링, FC, 활성화 함수, 역전파, 가중치 업데이트, 메모리 관리 등 모든 핵심 연산을 포함합니다. |
| **`utils.cpp`** | 바이너리 가중치 파일 로드, 다차원 배열 메모리 할당/해제, 필터 전치(transpose) 등 유틸리티 함수를 제공합니다. |
| **`alexnet/alexnet.hpp`** | `CNN` 클래스를 상속하여 AlexNet 고유의 레이어 구조(필터, 바이어스, 그래디언트 등)를 정의합니다. |
| **`alexnet/alexnet.cpp`** | 네트워크 생성(`create_network`), 가중치 설정(`set_training`), 학습 루프(`training_epoch`, `training_batch`), 메모리 해제(`destroy_network`), 가중치 저장(`save_weight_binary`)을 구현합니다. |

---

## 🔧 빌드 및 실행

### 사전 요구사항

- **C++ 컴파일러**: C++11 이상 지원 (g++)
- **CMake**: 3.8.1 이상
- **OpenCV**: OpenCV 4.x (이미지 읽기 및 전처리에 사용)
- **nlohmann/json**: `json.hpp` 헤더 파일 (프로젝트 루트에 배치)

### 빌드

```bash
mkdir build && cd build
cmake ..
make
```

### 실행

```bash
./alexnet
```

---

## 🐍 Python 유틸리티

`python/` 디렉토리에는 PyTorch와 C++ 간의 가중치 변환을 위한 스크립트가 포함되어 있습니다.

### 사전 요구사항 (Python)

- Python 3.x
- PyTorch, torchvision
- OpenCV (`cv2`)
- NumPy, tqdm

### 가중치 워크플로우

```
PyTorch Pretrained Model
        │
        ▼
 get_init_weight.py     ──→  바이너리 가중치 파일(.bin) 생성
        │
        ▼
  C++ AlexNet 학습       ──→  학습된 가중치 저장 (.bin)
        │
        ▼
set_trained_weight.py   ──→  학습된 가중치를 PyTorch로 로드하여 검증
```

| 스크립트 | 용도 |
|----------|------|
| `pytorch_network.py` | PyTorch AlexNet 모델 클래스 정의 (C++ 구현과 동일한 구조) |
| `get_init_weight.py` | PyTorch 모델의 가중치를 바이너리 파일(`.bin`)로 추출 |
| `set_trained_weight.py` | 바이너리 가중치를 PyTorch 모델에 로드하여 추론 결과 검증 |
| `pytorch_train_test.py` | PyTorch 환경에서의 학습/테스트 (비교 기준용) |

### 바이너리 가중치 파일 형식

가중치 파일은 다음과 같은 네이밍 컨벤션을 따릅니다:

- **Conv 필터**: `{output_ch}-{input_ch}-{height}-{width}.bin` (예: `64-3-11-11.bin`)
- **FC 가중치**: `{output}-{input}.bin` (예: `4096-9216.bin`)
- **Bias**: `{size}-{index}b.bin` (예: `64-0b.bin`)

모든 가중치는 **32-bit float (little-endian)** 형식으로 저장됩니다.

---

## 🧠 구현 상세

### Forward Pass
1. 입력 이미지 (224×224×3)를 읽어 RGB 채널별로 정규화 (mean=0.5, std=0.5)
2. 5개의 합성곱 레이어 + 3개의 맥스 풀링 레이어를 순차 통과
3. Flatten 후 3개의 완전 연결 레이어 통과
4. Softmax를 통해 1000개 클래스에 대한 확률 출력

### Backward Pass
1. Cross-Entropy Loss의 그래디언트 계산
2. FC 레이어 → Conv 레이어 순으로 역전파
3. 풀링 레이어는 max 위치 정보를 이용하여 그래디언트를 전달
4. SGD(Stochastic Gradient Descent)를 통한 가중치 업데이트

### 멀티스레드 지원
- C++ `<thread>` 라이브러리를 활용한 멀티스레드 연산 지원 (`-pthread` 플래그)

---

## 📋 참고사항

- 이 프로젝트는 **학습 및 교육 목적**으로 작성되었으며, 프로덕션 환경에서의 사용은 권장하지 않습니다.
- 메모리 관리는 수동 `new/delete`로 처리되므로, 대규모 데이터셋에서는 메모리 사용에 주의가 필요합니다.
- 가중치 초기화는 PyTorch의 사전 학습된 AlexNet 모델(`alexnet-owt-4df8aa71.pth`)을 사용합니다.

---

## 📄 라이선스

이 프로젝트는 개인 학습 프로젝트입니다.

---

## ✍️ 작성자

- **wonki hong** (2020-02-11 ~)
