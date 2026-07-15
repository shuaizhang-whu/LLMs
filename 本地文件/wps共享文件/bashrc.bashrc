# ~/.bashrc: 在非登录 shell 中由 bash(1) 执行
# 参考示例文件位置: /usr/share/doc/bash/examples/startup-files (在 bash-doc 包中)

# 不在历史记录中保存重复行或以空格开头的行
# 更多选项请参考 bash(1)
HISTCONTROL=ignoreboth

# 追加到历史文件，而不是覆盖它
shopt -s histappend

# 设置历史记录长度，参见 bash(1) 中的 HISTSIZE 和 HISTFILESIZE
HISTSIZE=1000
HISTFILESIZE=2000

# 每次命令执行后检查窗口大小，必要时更新 LINES 和 COLUMNS 的值
shopt -s checkwinsize

# 如果设置，路径扩展中的 "**" 模式将匹配所有文件和零个或多个目录及子目录
#shopt -s globstar

# 让 less 命令对非文本文件更友好，参见 lesspipe(1)
[ -x /usr/bin/lesspipe ] && eval "$(SHELL=/bin/sh lesspipe)"

# 设置标识 chroot 环境的变量（用于下面的提示符）
if [ -z "${debian_chroot:-}" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

# 设置美化的提示符（默认无颜色，除非我们明确需要颜色）
case "$TERM" in
    xterm-color|*-256color) color_prompt=yes;;
esac

# 如果终端支持，取消注释以启用彩色提示符
# 默认关闭以避免分散用户注意力：终端窗口的焦点应该在命令输出上，而不是提示符
force_color_prompt=yes

if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
	# 支持颜色显示；假定符合 Ecma-48 标准
	# (ISO/IEC-6429)。(不支持的情况极为罕见，此类情况倾向于支持 setf 而非 setaf)
	color_prompt=yes
    else
	color_prompt=
    fi
fi

if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
unset color_prompt force_color_prompt

# 如果是 xterm，设置标题为 user@host:dir
case "$TERM" in
xterm*|rxvt*)
    PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
    ;;
*)
    ;;
esac

# 启用 ls 和 grep 的彩色支持，并添加便捷别名
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# GCC 警告和错误的彩色显示
#export GCC_COLORS='error=01;31:warning=01;35:note=01;36:caret=01;32:locus=01:quote=01'

# 更多 ls 别名
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

# 为长时间运行的命令添加 "alert" 别名。使用方法：
#   sleep 10; alert
alias alert='notify-send --urgency=low -i "$([ $? = 0 ] && echo terminal || echo error)" "$(history|tail -n1|sed -e '\''s/^\s*[0-9]\+\s*//;s/[;&|]\s*alert$//'\'')"'

# 别名定义
# 你可以将所有自定义添加放到单独的文件中，如 ~/.bash_aliases，
# 而不是直接添加到这里。参见 bash-doc 包中的 /usr/share/doc/bash-doc/examples。

if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi

# 启用可编程补全功能
# 如果已在 /etc/bash.bashrc 和 /etc/profile 中启用，则无需在此启用
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi

# >>> conda 初始化 >>>
# !! 此块中的内容由 'conda init' 管理 !!
__conda_setup="$('/data/zhangs/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/data/zhangs/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/data/zhangs/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/data/zhangs/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda 初始化结束 <<<
export PATH=/usr/local/cuda-11.8/bin:$PATH

if [ -z "$VSCODE_IPC_HOOK_CLI" ]; then
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
fi

# ========================================
# 自动保存和恢复工作目录及 conda 环境
# ========================================

# 定义保存文件路径
LAST_DIR_FILE="$HOME/.last_directory"
LAST_ENV_FILE="$HOME/.last_conda_env"

# 保存当前目录的函数
save_current_dir() {
    pwd > "$LAST_DIR_FILE"
}

# 保存当前 conda 环境的函数
save_current_env() {
    if [ -n "$CONDA_DEFAULT_ENV" ]; then
        echo "$CONDA_DEFAULT_ENV" > "$LAST_ENV_FILE"
    fi
}

# 在每次 cd 命令后保存目录
cd() {
    builtin cd "$@" && save_current_dir
}

# 在 shell 退出时保存当前状态
trap 'save_current_dir; save_current_env' EXIT

# 启动时恢复上次的目录
if [ -f "$LAST_DIR_FILE" ]; then
    LAST_DIR=$(cat "$LAST_DIR_FILE")
    if [ -d "$LAST_DIR" ]; then
        builtin cd "$LAST_DIR" 2>/dev/null
    fi
fi

# 启动时恢复上次的 conda 环境
if [ -f "$LAST_ENV_FILE" ]; then
    LAST_ENV=$(cat "$LAST_ENV_FILE")
    if [ -n "$LAST_ENV" ] && [ "$LAST_ENV" != "base" ]; then
        # 检查环境是否存在
        if conda env list | grep -q "^$LAST_ENV "; then
            conda activate "$LAST_ENV" 2>/dev/null
        fi
    fi
fi

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

# Claude Code proxy
pgrep -f "pproxy.*8118" > /dev/null || pproxy -l http://:8118 -r socks5://127.0.0.1:7898 &
export HTTP_PROXY=http://127.0.0.1:8118
export HTTPS_PROXY=http://127.0.0.1:8118
