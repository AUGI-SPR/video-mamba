import pickle

for i in range(10, 81):
    with open(
        f"/data2/local_datasets/cholec80/features/resnet/video{i}.pkl", "rb"
    ) as f:
        data = pickle.load(f)
    print(data.shape)


# for i in range(10, 81):
#     # .pkl 파일 경로
#     resnet_file_path = f"/data2/local_datasets/cholec80/features/resnet/video{i}.pkl"
#     with open(resnet_file_path, "rb") as f:
#         data = pickle.load(f)
#     print(data[0].shape)
#     data = data[0]
#     lovit_file_path = f"/data2/local_datasets/cholec80/features/lovit/video{i}.pkl"
#     with open(lovit_file_path, "rb") as f:
#         lovit = pickle.load(f)
#     print(lovit.shape)
#     # 2. 배열의 차원을 (feature_dimension, size, 1, 1)로 변경
#     # 예를 들어, 기존 data shape이 (size, feature_dimension)인 경우
#     reshaped_data = data.reshape(data.shape[1], data.shape[0], 1, 1)

#     # 4. 변환된 배열을 다시 pkl 파일로 저장
#     output_path = resnet_file_path
#     with open(output_path, "wb") as f:
#         pickle.dump(reshaped_data, f)

#     with open(output_path, "rb") as f:
#         data = pickle.load(f)
#     print(data.shape)

# for i in range(10, 81):
#     # .pkl 파일 경로
#     resnet_file_path = f"/data2/local_datasets/cholec80/features/resnet/video{i}.pkl"
#     lovit_file_path = f"/data2/local_datasets/cholec80/features/lovit/video{i}.pkl"
#     # 기존 cholec.pkl 파일에서 데이터를 불러오기
#     with open(resnet_file_path, "rb") as f:
#         resnet = pickle.load(f)

#     with open(lovit_file_path, "rb") as f:
#         lovit = pickle.load(f)

#     print(resnet.shape)
#     print(lovit.shape)

import pickle
import numpy as np

# # 1. pkl 파일 불러오기
# for i in ["01", "26", "51", ]
#     input_path = f"/data2/local_datasets/cholec80/features/resnet/video01.pkl"
#     with open(input_path, "rb") as f:
#         data = pickle.load(f)
#     print(data.shape)
#     # 3. 마지막 값을 제거하여 (feature_dimension, size-1, 1, 1)로 변환
#     reshaped_data = data[:, :-1, :, :]

#     # 4. 변환된 배열을 다시 pkl 파일로 저장

#     with open(input_path, "wb") as f:
#         pickle.dump(reshaped_data, f)

#     with open(input_path, "rb") as f:
#         data = pickle.load(f)
#     print(data.shape)

#     print(f"Reshaped array saved to {input_path}")


# resnet_path = f"./video26.pkl"
# lovit_path = f"/data2/local_datasets/cholec80/features/lovit/video26.pkl"
# with open(resnet_path, "rb") as f:
#     resnet = pickle.load(f)
# with open(lovit_path, "rb") as f:
#     lovit = pickle.load(f)

# resnet = resnet[2][:].tolist()
# print(len(resnet))
# # Step 1: 매핑 딕셔너리 정의
# mapping_dict = {
#     "Preparation": 0,
#     "CalotTriangleDissection": 1,
#     "ClippingCutting": 2,
#     "GallbladderDissection": 3,
#     "GallbladderPackaging": 4,
#     "CleaningCoagulation": 5,
#     "GallbladderRetraction": 6,
# }


# # Step 2: 파일 읽기 및 숫자 배열로 변환
# def convert_txt_to_numbers(input_file_path):
#     number_array = []
#     with open(input_file_path, "r") as file:
#         for line in file:
#             line = line.strip()  # 줄 끝의 공백 제거
#             if line in mapping_dict:
#                 number_array.append(mapping_dict[line])  # 매핑된 숫자 추가
#             else:
#                 number_array.append(None)  # 매핑되지 않는 문자열의 경우 None 추가

#     return number_array


# # 예시: 텍스트 파일 경로
# input_file_path = "/data2/local_datasets/cholec80/groundtruth/video26.txt"

# # 숫자 배열로 변환
# number_array = convert_txt_to_numbers(input_file_path)

# # 결과 출력
# print(len(number_array))
