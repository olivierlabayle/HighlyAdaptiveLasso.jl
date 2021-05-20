using RCall

R"install.packages('hal9001', repos='https://cloud.r-project.org/')"

print(read("deps/build.log", String))