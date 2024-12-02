import numpy as np

if __name__=="__main__":
    test = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ]
    
    test = np.array(test)
    
    print(test)
    print(np.mean(test, axis=0)) # axis, 平均を行か列かでとるやつらしい
    # print(test[:, :1])
    # print(test[1, :2])