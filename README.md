# shopping-classification

카카오에서 개최하는 `쇼핑몰 상품 카테고리 분류` 대회 참가용 코드입니니다. (코드는 python2.7, pytorch 기준으로 작성되었습니다.)

## 로직 설명
mlp_model 은 한 층 짜리 선형 모델로, 하위 카테고리 예측 시 상위 카테고리 예측 값의 정보를 각 카테고리의 embedding vector로 입력 받게 됩니다.
lstm_model 은 one-to-many LSTM 모델로, text feature가 들어가면 "대/중/소/세" 분류가 output으로 나오게 됩니다.

## 라이선스

This software is licensed under the Apache 2 license, quoted below.

Copyright 2018 Kakao Corp. http://www.kakaocorp.com

Licensed under the Apache License, Version 2.0 (the “License”); you may not use this project except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
