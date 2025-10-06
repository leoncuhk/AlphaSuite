# 安装依赖包目录

此目录用于存放手动下载的依赖包（如 wheel 文件）。

## macOS 用户说明

对于 macOS 用户，推荐使用 **Homebrew** 安装系统依赖，无需手动下载 wheel 文件：

```bash
# 安装 TA-Lib C 库
brew install ta-lib

# 安装 PostgreSQL
brew install postgresql@15
```

安装完成后，可以直接使用 `uv pip install -r requirements.txt` 安装所有 Python 依赖。

## Windows 用户说明

Windows 用户可能需要手动下载 TA-Lib 的 wheel 文件：

1. 访问 [TA-Lib Wheels](https://github.com/cgohlke/talib-build/releases)
2. 下载对应 Python 版本和系统架构的 wheel 文件
3. 将 wheel 文件保存到此目录
4. 安装：`pip install install/TA_Lib-0.x.x-cpXX-cpXX-win_amd64.whl`

## 其他依赖

如遇到其他难以安装的依赖包，也可以下载对应的 wheel 文件到此目录进行安装。