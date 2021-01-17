include("Tools.jl");
include("PlotTools.jl");

iris_data, tags = read_iris("data/iris.data");
new_data = tsne(normalize(iris_data), verbose=true);

iris_setosa = new_data[tags .== "Iris-setosa", :];
iris_versicolor = new_data[tags .== "Iris-versicolor", :];
iris_virginica = new_data[tags .== "Iris-virginica", :];

fig = scatterPL(iris_setosa[:, 1], iris_setosa[:, 2], color=:red);
scatterPL(iris_versicolor[:, 1], iris_versicolor[:, 2], fig, color=:black);
scatterPL(iris_virginica[:, 1], iris_virginica[:, 2], fig, color=:orange)
