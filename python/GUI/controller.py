# export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0

import View



if __name__ == '__main__':
    v = View.View()