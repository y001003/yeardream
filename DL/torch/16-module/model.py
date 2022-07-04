import torch
import torch.nn as nn

# nn.Module을 상속받은 class 정의
class Block(nn.Module):
    '''
    nn.Module을 상속받은 class의 경우 보통 2가지 def를 override한다.
    init, forward 함수를 아래와 같은 형태로 override하여 사용한다.
    '''

    def __init__(self,
                 input_size,
                 output_size,
                 use_batch_norm=True,
                 dropout_p=.4):
        self.input_size = input_size
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p
        
        # 부모 클래스 nn.Module 실행
        super().__init__()

        # use_batch_norm이 True 일 경우 배치 정규화 BatchNorm1d, False 인 경우 Dropout
        def get_regularizer(use_batch_norm, size):
            return nn.BatchNorm1d(size) if use_batch_norm else nn.Dropout(dropout_p)
        
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size), # input_size를 output_size로 바꾸어주는 역할
            nn.LeakyReLU(), 
            get_regularizer(use_batch_norm, output_size) # 활성화 함수를 거친 후에 배치 정규화
        )
    
    # 실질적으로 연산이 이루어지게 하는 코드, 따로 forward 함수를 실행시키지 않아도 __init__에서 정의한 block을 y로 맵핑하면서 실행된다.
    def forward(self, x):
        # |x| = (batch size, input_size)
        y = self.block(x) # 
        # |y| = (batch_size, output_size)
        return y
    

class ImageClassifier(nn.Module):
    
    def __init__(self,
                 input_size, # 들어가는 인풋 레이어 갯수 (MNIST 784 개)
                 output_size, # 클래스별 로그확률 갯수(MNIST 10개)
                 hidden_sizes = [500, 400, 300, 200, 100], # 중간 레이어의 사이즈
                 use_batch_norm = True,
                 dropout_p=.3):
        
        # 부모 클래스 nn.Module 실행
        super().__init__()
        
        # 예외처리 hidden_size가 없으면 안된다.
        assert len(hidden_sizes) > 0, "You need to specify hidden layers"

        last_hidden_size = input_size
        blocks = []
        for hidden_size in hidden_sizes: # 784 -> 500 -> 400->300->200->100 -> 10개
            blocks += [
                #위에서 정의한 class 불러온다.
                Block( # 히든레이어 층 만들기
                    last_hidden_size, # input size
                    hidden_size, # output size
                    use_batch_norm,
                    dropout_p
                )
            ]
            last_hidden_size = hidden_size
        
        # 출력층
        self.layers = nn.Sequential(
            *blocks,
            nn.Linear(last_hidden_size, output_size),
            nn.LogSoftmax(dim=-1),
        )
    
    def forward(self, x):
        # |x| = (batch size, input_size)
        y = self.layers(x) # 
        # |y| = (batch_size, output_size)
        return y