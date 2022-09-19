
import pandas as pd
import json

# Bütün satırların ve sütunların gösterilmesi
pd.set_option('display.max_columns', None);
pd.set_option('display.max_rows', None);
pd.set_option('display.width', 500)

# Virgülden sonra gösterilecek basamak sayısı
pd.set_option('display.float_format', lambda x: '%.2f' % x)


json_data = json.load(open("outputs/model_info_data.json", 'r'))

print(pd.json_normalize(json_data['data'], max_level=0).sort_values("date", ascending=False))