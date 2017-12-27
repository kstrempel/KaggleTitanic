import pandas as pd


def transform(source, target, train):
    d = pd.read_csv(source)
    ## replace or filter missing information
    d.loc[(d.Sex == 'female') & (d.Age.isna()) & (d.Name.str.contains('Miss.')),'Age'] = 21
    d.loc[(d.Sex == 'female') & (d.Age.isna()) & (d.Name.str.contains('Mrs.')),'Age'] = 35
    d.loc[(d.Sex == 'female') & (d.Age.isna()) & (d.Name.str.contains('Ms.')),'Age'] = 35
    d.loc[(d.Sex == 'male') & (d.Age.isna()),'Age'] = 33
    ## drop some columns
    columns = ['Sex','Age','Pclass','SibSp','Parch']
    if train:
        columns = ['Survived'] + columns
    columns = ['PassengerId'] + columns
    d = d[columns]
    ## transform sex to number 0 male 1 femal
    sex_column = d['Sex']
    d.loc[sex_column == 'female', 'Sex'] = 1
    d.loc[sex_column == 'male', 'Sex'] = 0
    d.to_csv(target, header=False, index=False)


if __name__ == '__main__':
    transform('train.csv','train_transformed.csv', train=True)
    transform('test.csv','test_transformed.csv', train=False)
