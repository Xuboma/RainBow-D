class Runner:
    # 重量可以输入，p是固定值 v是运动速度 A也设置为固定值 c也是固定值
    weight = 0.0
    p = 1.29  # 空气密度
    v = 0.0    # 跑步速度
    A = 0.8   # 人的迎风面积
    c = 0.865   # 空气阻力系数
    distance = 0.0 # 人跑步的总距离
    person_w = 110.0  # 每个人每s身体代谢消耗110J的能量
    distance = 0.0

    def __init__(self, weight, distance):
        self.weight = weight
        self.distance = distance

    # 跑步每秒克服阻力消耗的能量函数
    def W_s(self, v):
        # 个人所受的风的阻力
        self.v = v
        F = 0.5*self.p*self.v*self.v*self.A*self.c
        S = self.v*1.0
        return F*S

    # 计算跑步克服阻力总共地消耗能量     也可用来后期的计算使用
    def total_w(self, distance):
        self.distance = distance
        return self.weight*self.distance*1.036*4185

def test():
    runner = Runner(60, 0.8)
    WS = runner.W_s(3.5)
    total_w2 = runner.total_w(0.8)
    print(WS, ' ', total_w2)
    print(11.1*(WS+110))


# test()
# print()
