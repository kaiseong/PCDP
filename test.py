import csv
 
state = {
    'ActualTCPPose': [-0.28676133, 0.28277331, 0.04765055, 1.36036411, 2.82111225, 0.01397401],
    'ActualTCPSpeed': [0., 0., 0., 0., 0., 0.],
    'ActualQ': [-1.04568321, -1.58212597, 2.28941321, -2.27508718, 4.691113, 4.5569334],
    'ActualQd': [0., 0., -0., 0., 0., 0.],
    'TargetTCPPose': [-0.28677046, 0.28275414, 0.04763933, 1.36047462, 2.82102614, 0.01385819],
    'TargetTCPSpeed': [0., 0., 0., 0., 0., 0.],
    'TargetQ': [-1.04566253, -1.58210576, 2.28940915, -2.27503575, 4.69115697, 4.55704234],
    'TargetQd': [0., 0., 0., 0., 0., 0.],
    'robot_receive_timestamp': 1745327190.4625356
}

# 파일에 쓰기
with open('tttest_state_output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # state를 문자열로 변환해서 하나의 셀에 넣기
    writer.writerow([str(state)])  # 대괄호 []로 리스트로 감싸야 한 row에 한 칸만 써짐
