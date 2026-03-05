from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

# --- MINIMALIST UNIFORM STYLE CONFIG ---
COLOR_PRIMARY = (30, 30, 30)      
COLOR_TEXT = (50, 50, 50)      
FONT_FAMILY = "DejaVu"          
BASE_FONT_SIZE = 11.0          

class SteppeDNAPDF(FPDF):
    def __init__(self):
        super().__init__()
        # Load Windows Arial fonts with UTF-8 support
        self.add_font(FONT_FAMILY, "", r"c:\Windows\Fonts\arial.ttf")
        self.add_font(FONT_FAMILY, "B", r"c:\Windows\Fonts\arialbd.ttf")
        self.add_font(FONT_FAMILY, "I", r"c:\Windows\Fonts\ariali.ttf")

    def header(self):
        if self.page_no() == 1:
            self.set_font(FONT_FAMILY, "B", BASE_FONT_SIZE + 3)
            self.set_text_color(*COLOR_PRIMARY)
            self.cell(0, 10, "SteppeDNA: Полное описание проекта", 
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
            self.set_draw_color(30, 30, 30)
            self.set_line_width(0.3)
            self.line(self.get_x(), self.get_y(), self.get_x()+190, self.get_y())
            self.ln(4)
            
            self.set_font(FONT_FAMILY, "", BASE_FONT_SIZE)
            self.set_text_color(*COLOR_TEXT)
            self.multi_cell(0, 6, "Простое руководство о том, как SteppeDNA использует Искусственный Интеллект для предсказания риска рака по генетическим мутациям. Этот документ объясняет точные размеры наборов данных, масштаб проекта, то, как ИИ принимает решения, и как мы гарантируем, что он не «списывает» на тестах.")
            self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font(FONT_FAMILY, "", BASE_FONT_SIZE - 2)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Описание проекта SteppeDNA | Страница {self.page_no()}", align="C")

    def add_section_header(self, text):
        self.ln(4)
        self.set_font(FONT_FAMILY, "B", BASE_FONT_SIZE + 1)
        self.set_text_color(*COLOR_PRIMARY)
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def add_sub_header(self, text):
        self.ln(2)
        self.set_font(FONT_FAMILY, "B", BASE_FONT_SIZE)
        self.set_text_color(*COLOR_PRIMARY)
        self.cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*COLOR_TEXT)

    def add_bullet(self, label, text):
        self.set_left_margin(15)
        self.set_text_color(*COLOR_PRIMARY)
        
        if label:
            self.set_font(FONT_FAMILY, "B", BASE_FONT_SIZE)
            self.write(6, f"- {label}: ")
            self.set_font(FONT_FAMILY, "", BASE_FONT_SIZE)
            self.set_text_color(*COLOR_TEXT)
            self.multi_cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            self.set_font(FONT_FAMILY, "", BASE_FONT_SIZE)
            self.set_text_color(*COLOR_TEXT)
            self.multi_cell(0, 6, f"- {text}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
        self.set_left_margin(10)
        
    def draw_placeholder_box(self):
        self.ln(6)
        start_x = self.get_x() + 10
        start_y = self.get_y()
        self.set_draw_color(180, 180, 180)
        self.set_line_width(0.2)
        
        width = 170
        height = 60
        
        self.rect(start_x, start_y, width, height)
        
        self.set_xy(start_x, start_y + (height/2) - 6)
        self.set_font(FONT_FAMILY, "", BASE_FONT_SIZE)
        self.set_text_color(120, 120, 120)
        self.cell(width, 12, "[ Вставьте сюда последний скриншот интерфейса SteppeDNA ]", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        self.set_xy(10, start_y + height + 6)

def generate_pdf():
    pdf = SteppeDNAPDF()
    pdf.set_margins(12, 15, 12)
    pdf.add_page()
    pdf.set_font(FONT_FAMILY, "", BASE_FONT_SIZE)
    pdf.set_text_color(*COLOR_TEXT)

    # SEC 1
    pdf.add_section_header("1. Что делает SteppeDNA?")
    pdf.multi_cell(0, 6, "Представьте человеческую ДНК как огромную инструкцию. Иногда в этой инструкции встречается «опечатка» (мутация). SteppeDNA рассматривает 5 конкретных «глав» (генов) этой инструкции, которые занимаются восстановлением ДНК: BRCA1, BRCA2, PALB2, RAD51C и RAD51D. \n\nКогда врач находит у пациента опечатку, он часто не знает, является ли она «Патогенной» (опасной, повышающей риск рака) или «Доброкачественной» (безвредной опечаткой). Лабораторные тесты для выяснения этого занимают месяцы. SteppeDNA использует Искусственный Интеллект, чтобы мгновенно «прочитать» опечатку и предсказать, опасна ли она.")
    
    pdf.add_sub_header("Размер и масштаб проекта")
    pdf.add_bullet("Всего файлов", "154 файла, содержащих данные, код сайта и логику ИИ.")
    pdf.add_bullet("Строк кода", "13 097 строк программы (в основном на Python).")
    pdf.add_bullet("Всего генов", "5 ключевых генов рака молочной железы и яичников.")
    pdf.add_bullet("Подсказки (Признаки)", "Для каждой отдельной мутации собирается 128 различных математических и биологических подсказок, чтобы помочь ИИ принять решение.")

    # SEC 2
    pdf.add_section_header("2. Честная оценка: Насколько хорош ИИ?")
    pdf.multi_cell(0, 6, "Прямо как студент на экзамене, ИИ иногда может «списать», если за ним не следить. Со временем мы усложняли тесты, чтобы увидеть, насколько действительно «умен» проект SteppeDNA.")
    
    pdf.add_sub_header("Эра 99% (Чит-код)")
    pdf.multi_cell(0, 6, "Изначально ИИ показал почти идеальную точность 99% на одном гене (BRCA2). Однако мы поняли, что давали ИИ результаты реального лабораторного теста (так называемый MAVE-score) в качестве одной из подсказок. ИИ не изучал биологию; он просто читал шпаргалку врача!")
    
    pdf.add_sub_header("Эра 73% (Реальный универсальный тест)")
    pdf.multi_cell(0, 6, "Мы удалили «чит-код» и заставили ИИ изучать все 5 генов одновременно (759 различных мутаций). В реальном мире ИИ столкнется с мутациями, которые ни один человек ранее не видел, поэтому мы строго тестировали его на «невидимых» данных. \n\nЧестная оценка точности составляет 73,6% (по метрике 'ROC-AUC'). В мире медицинского ИИ предсказание сложной биологии рака в 5 совершенно разных генах с точностью 73,6% без шпаргалок — это очень сильный, реалистичный результат, показывающий, что ИИ действительно усвоил естественные правила жизни.")

    # SEC 3
    pdf.add_section_header("3. Как мы предотвращаем «заучивание» (Переобучение)")
    pdf.multi_cell(0, 6, "ИИ часто пытается вызубрить учебник, вместо того чтобы понять суть. Мы встроили программные блокировки, чтобы остановить это (предотвращение «Утечки данных»):")
    pdf.add_bullet("Создание тренировочных вопросов (SMOTE)", "Опасные мутации редки. Чтобы помочь ИИ тренироваться, мы генерируем «синтетические» поддельные опасные мутации. Но критически важно, что мы делаем это ТОЛЬКО во время «учебы» (обучения). Если бы мы оставили поддельные примеры в финальном тесте, ИИ получил бы искусственно завышенную оценку.")
    pdf.add_bullet("Скрытие теста", "Мы тестируем ИИ 5 раз отдельно, используя метод «Кросс-валидация». В каждом тесте от ИИ полностью скрывают тестовые вопросы, пока он учится.")
    pdf.add_bullet("Удаление имен", "Мы скрываем точные названия мутаций от ИИ, чтобы он не мог запомнить, что «Мутация X - плохая». Он должен смотреть только на химию.")

    # SEC 4
    pdf.add_section_header("4. 128 подсказок: Как ИИ принимает решения")
    pdf.multi_cell(0, 6, "Чтобы сделать предсказание, SteppeDNA собирает 128 уникальных подсказок об «опечатке». Вот некоторые из самых важных, объясненные просто:")
    
    pdf.add_sub_header("Эволюционные подсказки (Изменяла ли это природа?)")
    pdf.add_bullet("PhyloP (Тест длиной в 400 млн лет)", "Мы проверяем ДНК 100 животных (от рыб до обезьян). Если участок ДНК не изменился за 400 миллионов лет, значит, Природа считает его идеально настроенным. Если мутация у человека изменяет именно это место, она почти наверняка опасна.")
    
    pdf.add_sub_header("Физические подсказки (Ломает ли это механизм?)")
    pdf.add_bullet("Размер и заряд", "Белки — это 3D-пазлы. Если вы замените крошечную деталь с отрицательным зарядом на огромную с положительным зарядом, вся структура может развалиться.")
    pdf.add_bullet("Спрятано ли это?", "Если опечатка происходит глубоко внутри ядра белка, куда не должна попадать вода, это вызывает более серьезные повреждения, чем на поверхности.")
    pdf.add_bullet("Расстояние до двигателя", "Белки связываются с ДНК, чтобы ремонтировать её. Мы рассчитываем, насколько физически близко опечатка находится к «рукам», захватывающим ДНК.")

    pdf.add_page()
    
    # SEC 5
    pdf.add_section_header("5. Откуда мы берем наши данные")
    pdf.multi_cell(0, 6, "SteppeDNA автоматически подключается к огромным глобальным базам данных для сбора подсказок:")
    pdf.add_bullet("ClinVar", "Глобальная база данных известных мутаций рака. Выступает в качестве наших «ответов» для обучения.")
    pdf.add_bullet("ESM-2 (Meta AI)", "Искусственный интеллект, созданный Meta, который работает как словарь-переводчик для белков. Дает 64 подсказки о «грамматике» мутации.")
    pdf.add_bullet("gnomAD", "База данных 800 000 здоровых людей. Если мутация очень часто встречается у здоровых людей, SteppeDNA делает уверенный вывод, что она не может вызывать тяжелый рак.")
    pdf.add_bullet("AlphaMissense", "Собственные предсказания от Google DeepMind. Мы просим Google дать второе мнение и загружаем его в нашу систему.")

    # SEC 6
    pdf.add_section_header("6. «Комитет» моделей ИИ")
    pdf.multi_cell(0, 6, "SteppeDNA не спрашивает только один ИИ. Приложение спрашивает комитет из двух разных типов ИИ и усредняет их голоса для надежности:")
    pdf.add_bullet("Дерево решений (XGBoost)", "Этот ИИ очень хорош в анализе сухих фактов (например, «распространено ли это среди здоровых людей?»). Он действует как логический мыслитель.")
    pdf.add_bullet("Нейронная сеть", "Этот ИИ работает больше как человеческий мозг. Он анализирует 64 «грамматические» подсказки от ИИ Meta, чтобы найти скрытые невидимые паттерны, которые простая математика не может увидеть.")
    pdf.add_bullet("Калибратор", "Математический фильтр, который берет сырые ощущения «да/нет» от ИИ и преобразует их в реалистичные, заслуживающие доверия проценты от 0 до 100%.")

    # SEC 7
    pdf.add_section_header("7. Медицинские правила и обратная трансляция")
    pdf.multi_cell(0, 6, "Отличительной чертой SteppeDNA является то, что платформа уважает правила врачей и биологию:")
    pdf.add_bullet("Стандартные правила врачей (ACMG)", "У врачей есть стандартные списки для классификации мутаций. SteppeDNA автоматически заполняет эти списки (например, отмечая, находится ли мутация слишком близко к области связывания с ДНК).")
    pdf.add_bullet("Обратная биология (Минус-цепь)", "Некоторые гены (например, BRCA2) записаны «задом наперед» на двойной спирали ДНК. SteppeDNA автоматически рассчитывает эту обратную математику за миллисекунды, переворачивая буквы ДНК так, чтобы ИИ получил правильную последовательность.")

    pdf.draw_placeholder_box()

    output_path = r"c:\Users\User\OneDrive\Desktop\Project explanation\v2_SteppeDNA_Deep_Dive_RU.pdf"
    try:
        pdf.output(output_path)
        print(f"Success! Minimalist Uniform B&W Layman Russian PDF generated at {output_path}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    generate_pdf()
