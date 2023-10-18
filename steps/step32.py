# No code
# Variable 클래스: 기존 코드의 np 관련 부분을 Variable 클래스로 래핑.
# Function 클래스: backward 메서드를 수정. 역전파로 계산 그래프를 만들도록 수정.
# 역전파 필요 없는 경우, 비활성 모드 추가 -> 메모리 최적화
