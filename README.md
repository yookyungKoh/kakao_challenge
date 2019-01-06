# shopping-classification

카카오에서 개최하는 `쇼핑몰 상품 카테고리 분류` 대회 참가용 코드입니다.
코드는 기존에 공개된 코드를 참고하여 작성되었습니다. (https://github.com/kakao-arena/shopping-classification)

## 실행 방법
기존 베이스라인 코드의 실행방법과 일치합니다.

0. 데이터 위치
    - 데이터의 위치는 코드가 실행되는 디렉토리의 상위 디렉토리로(../)  가정되어 있습니다.

1. 'python data.py make_db train`
    - 학습에 필요한 데이터셋을 생성합니다. (h5py 포맷) dev, test도 동일한 방식으로 생성할 수 있습니다.
    - 위 명령어를 수행하면 `train` 데이터의 80%는 학습, 20%는 평가로 사용되도록 데이터가 나뉩니다.
    - 'test' 데이터셋 예시:  'python data.py make_db test ./data/test --train_ratio=0.0'

2. 'python classifier.py train ./data/train ./model/train`
    - `./data/train`에 생성한 데이터셋으로 학습을 진행합니다.
    - 완성된 모델은 `./model/train`에 위치합니다.

3. 'python classifier.py predict ./data/train ./model/train ./data/test/ dev predict.tsv`
    - 위 실행 방법에서 생성한 모델로 `test` 데이터셋에 대한 예측 결과를 생성합니다.


## 로직 설명
input으로는 텍스트와과 이미지 feature를 받습니다.  텍스트는 상품명에 속하는 단어 unigram과 음절 단위의 character unigram으로 이루어진 sequence이고, 이미지 feature는 제공된 ResNet50 모델의 출력 값입니다.
대, 중, 소, 세 카테고리에 대한 context vector를 각각 만들고 이를 text feature로 통합합니다. text와 image feature에 대한 multimodal weight을 구해 각각 할당합니다.
각 weight이 곱해진 text / image feature vector를 합친 후, 기본 선형 모델에 residual connection layer 2개를 거쳐 최종 output이 출력됩니다.
output은 계층 구분 없는 "대>중>소>세"를 표현한 class에 대한 확률값입니다.

## 라이선스

This software is licensed under the Apache 2 license, quoted below.

Copyright 2018 Kakao Corp. http://www.kakaocorp.com

Licensed under the Apache License, Version 2.0 (the “License”); you may not use this project except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
