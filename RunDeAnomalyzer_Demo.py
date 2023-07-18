from DeAnomalyzer import DeAnomalyzer



if __name__ == '__main__':
    print("DeAnomalyzer\n\n")
    
    da = DeAnomalyzer("IF", "Sklearn")
    da.fit("Dataset/ar1.csv")
    print(da.univariate_ari)
    


