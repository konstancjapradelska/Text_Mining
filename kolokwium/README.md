## Python version

- [**Python**](https://www.python.org/downloads/release/python-392/) - 3.9.2

## Setup Virtualenv on Windows

```console
py -3.9 -m pip install virtualenv
py -3.9 -m virtualenv venv
venv\Scripts\activate
venv\Scripts\pip3 install -r requirements.txt
venv\Scripts\pip3 install wordcloud-1.8.1-cp39-cp39-win_amd64.whl
venv\Scripts\python.exe -m nltk.downloader stopwords
```
Next configure your IDE. If you are using PyCharm `->` [Help](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html#env-requirements) <br>
`Add interpreter -> Existing enviroment -> Choose interpreter (.\venv\Scripts\python.exe)`

If any problem with activating venv try: 

```
For Windows 11, Windows 10, Windows 7, Windows 8, Windows Server 2008 R2 or Windows Server 2012, run the following commands as Administrator:

x86 (32 bit)
Open C:\Windows\SysWOW64\cmd.exe
Run the command: powershell Set-ExecutionPolicy RemoteSigned

x64 (64 bit)
Open C:\Windows\system32\cmd.exe
Run the command: powershell Set-ExecutionPolicy RemoteSigned
```

## Running Application
To run application open terminal in root folder and type:

- `venv\Scripts\python .\main.py`