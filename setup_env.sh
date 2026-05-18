#!/bin/bash
# /home/mahe/your_project_folder/setup_env.sh

export http_proxy=http://proxy.cc.yamaguchi-u.ac.jp:8080
export https_proxy=http://proxy.cc.yamaguchi-u.ac.jp:8080
npm config set proxy http://proxy.cc.yamaguchi-u.ac.jp:8080
npm config set https-proxy http://proxy.cc.yamaguchi-u.ac.jp:8080

# 定义项目内的快捷别名
alias proxy-off="unset http_proxy && unset https_proxy && npm config delete proxy && npm config delete https-proxy && echo '❌ 代理已关闭'"

echo '✅ 实验室 Node4 代理已开启'