'''
Возвращает csv-массив данных по КИП + ПАК + ВАК приведенных по времени
'''
import pandas as pd

from typing import Optional, Any
from pathlib import Path
import sys
import subprocess

UPLOADS = Path(__file__).resolve().parent / "data"
CURRENT_DIR = Path(__file__).resolve().parent

def get_all_data(avt_tags: pd.DataFrame, quality_tags: pd.DataFrame, pak_tags: pd.DataFrame) -> pd.DataFrame:
    '''
    Функция объединения данных
    '''

    common = avt_tags.columns.intersection(quality_tags.columns)
    quality_tags = quality_tags.drop(columns=common) # Удаляем дублирующие с АВТ колонки


    df = avt_tags.join(quality_tags, how="outer").join(pak_tags, how="outer")

    return df.sort_index()

def get_csv_tags(path_to_data: str) -> pd.DataFrame:

    '''
    Функция получения тегов АВТ
    '''
    df = pd.read_csv(
        path_to_data,
        sep=",",
        parse_dates=["date"],
        index_col="date",
    )

    
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    
    df = df.astype(float)

    return df

def run_quality_formulas(current_path: Path) -> Any:
    '''
    Функция запуска расчета ВАК (запуск quality_formulas.py)
    '''
    script = current_path / "quality_formulas.py"

    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=script.parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print("\nКод завершения:", e.returncode)
        print("\nstdout:", e.stdout)
        print("\nstderr:", e.stderr)
        return None
    else:   
        return result

def get_pak_tags(path_to_data: str) -> pd.DataFrame:

    '''
    Функция получения тегов ПАК
    '''
    raw = pd.read_excel(path_to_data, header=None)

  
    name1 = str(raw.iat[0, 0]).strip()
    name2 = str(raw.iat[0, 3]).strip()

    def extract(col_date: int, col_val: int, name: str) -> pd.DataFrame:
       
        sub = raw.iloc[2:, [col_date, col_val]].copy()
        sub.columns = ["date", name]

        
        sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
        sub = sub.dropna(subset=["date"])


        sub[name] = pd.to_numeric(sub[name], errors="coerce")

        return sub.set_index("date").sort_index()

    df1 = extract(col_date=0, col_val=1, name=name1)   
    df2 = extract(col_date=3, col_val=4, name=name2)   
   
    df = df1.join(df2, how="outer").sort_index()

    return df

def main():
    print('\nНачало получения данных')

    output_file = UPLOADS / "output.csv"

    if output_file.exists():
        print('\nНайден уже существующий файл с данными.')

        while True:
            no = input("\nВыполнить перерасчет? (y/n) [n]:")

            if no.strip().lower() in ("n", ""):
                return
            elif no.strip().lower() == 'y':
                break
            else:
                print('\nНекорректный ввод')

    print('\nПолучение тегов АВТ')
    
    avt_tags = get_csv_tags(f"{UPLOADS}/avt_tags.csv")

    print('\nТеги АВТ успешно получены\n\nЗапущен расчет ВАК')

    output_file = UPLOADS / "tags_with_quality.csv"

    if output_file.exists():
            print('\nНайден уже существующий файл с ВАК тегами.')
    
            while True:
                no = input("\nВыполнить перерасчет? (y/n) [n]:")
    
                if no.strip().lower() in ("n", ""):
                    break

                elif no.strip().lower() == 'y':

                    res = run_quality_formulas(CURRENT_DIR)

                    if res is None:
                        print('\nОшибка расчета')
                        return
                    else:
                        print('\nРасчет ВАК закончен. Получение значений')
                    break

                else:
                    print('\nНекорректный ввод')
       
    quality_tags = get_csv_tags(f"{UPLOADS}/tags_with_quality.csv")

    print('\nЗначения ВАК успешно получены\n\nПолучение значений ПАК')

    pak_tags = get_pak_tags(f"{UPLOADS}/Выгрузка ПАК 01.01.2023 - н.в_.xlsx")

    print('\nЗначения ПАК успешно получены. Итоговый массив будет сохранен в файл data/output.csv')

    df = get_all_data(avt_tags, quality_tags, pak_tags)

    df.to_csv(f"{UPLOADS}/output.csv", sep=",")

if __name__ == '__main__':
    main()