import pandas as pd


def transform(source, target, train):
    data = pd.read_csv(source)
    ## filter out missing information
    if train:
        data = data.dropna(subset=['Age'], how='any')
    else:
        data.loc[data['Age'].isna(),'Age'] = 20
    ## drop some columns
    columns = ['Sex','Age','Pclass']
    if train:
        columns = ['Survived'] + columns
    columns = ['PassengerId'] + columns
    data = data[columns]
    ## transform sex to number 0 male 1 femal
    sex_column = data['Sex']
    data.loc[sex_column == 'female', 'Sex'] = 1
    data.loc[sex_column == 'male', 'Sex'] = 0
    data.to_csv(target, header=False, index=False)    


if __name__ == '__main__':
    transform('train.csv','train_transformed.csv', train=True)
    transform('test.csv','test_transformed.csv', train=False)    