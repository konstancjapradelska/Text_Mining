import re

# 1.1 zadanie
liczby = "Dzisiaj mamy 4 stopnie na plusie, 1 marca 2022 roku"
result = re.sub('\d', '', liczby)
print(result)

# 1.2 zadanie
znaki_html = '<div><h2>Header</h2> <p>article<b>strong text</b> <a href=\"\">link</a></p></div>'
result = re.sub('[<>/]', '', znaki_html)
print(result)

# 1.3 zadanie
znaki_interpunkcyjne = 'Lorem ipsum dolor sit amet, consectetur; adipiscing elit. ' \
                       'Sed eget mattis sem. Mauris egestas erat quam, ut faucibus eros congue et.' \
                       ' Inblandit, mi eu porta; lobortis, tortor nisl facilisis leo, ' \
                       'at tristique augue risuseu risus.'
result = re.sub('[.,;]', '', znaki_interpunkcyjne)
print(result)

# 2 zadanie
hashtagi = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. ' \
           'Sed #texting eget mattis sem. Mauris #frasista egestas erat ' \
           '#tweetext quam, ut faucibus eros #frasier congue et. In blandit, ' \
           'mi eu porta lobortis, tortor nisl facilisis leo, at tristique #frasistas augue risus eu risus.'
result = re.findall(r"#(\w+)", hashtagi)
print(result)

# 3 zadanie
emoticons = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. :)' \
            ' Sed #texting eget mattis sem. Mauris #frasista egestas erat ' \
            '#tweetext quam,:> ut faucibus eros #frasier congue et. In blandit, ' \
            'mi eu porta lobortis, tortor nisl facilisis leo, at tristique ;( #frasistas augue risus eu risus.'
result = re.findall(':\)|;\)|;\(|:>|:<|;<|:-\)|;-\)', emoticons)
print(result)
