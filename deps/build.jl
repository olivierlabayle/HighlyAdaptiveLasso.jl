using RCall

R"""
install.packages('devtools', repos='http://cran.us.r-project.org')
require(devtools)
install_version("hal9001", version = "0.4.1", repos = "http://cran.us.r-project.org")
"""