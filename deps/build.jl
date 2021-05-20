using RCall

println(ENV["R_HOME"])

R"install.packages('hal9001', repos='https://cloud.r-project.org/')"