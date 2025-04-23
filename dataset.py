import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')
sns.set_palette("deep")

df = pd.read_csv('titanic.csv')

df = df.dropna(subset=['Survived', 'Pclass', 'Sex'])
df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(df['Age'].median())
df['Survived'] = df['Survived'].astype(int)
df['Pclass'] = df['Pclass'].astype(int)

df['Surname'] = df['Name'].str.split(',').str[0].str.strip()

# Количество выживших
def survival_count():
	survival_counts = df['Survived'].value_counts()
	total = len(df)
	survival_percent = survival_counts / total * 100
	labels = ['Погибли', 'Выжили']
	plt.figure(figsize=(8, 8))
	plt.pie(survival_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
	plt.title('Количество выживших на Титанике')
	plt.axis('equal')
	plt.tight_layout()
	plt.show()
	survived = survival_counts[1] if 1 in survival_counts else 0
	died = survival_counts[0] if 0 in survival_counts else 0
	print("\nСтатистика выживших:")
	print(f"Всего пассажиров: {total}")
	print(f"Выжили: {survived} ({survival_percent[1]:.1f}%)")
	print(f"Умерли: {died} ({survival_percent[0]:.1f}%)")

# Количество выживших по полу
def survival_by_sex():
	survival_rate = df.groupby('Sex')['Survived'].mean().reset_index()
	sex_counts = df['Sex'].value_counts()
	survival_counts = df.groupby(['Sex', 'Survived']).size().unstack().fillna(0)
	plt.figure(figsize=(8, 6))
	sns.barplot(x='Sex', y='Survived', data=survival_rate)
	plt.title('Количество выживших по полу')
	plt.xlabel('Пол')
	plt.ylabel('Выжившие')
	plt.ylim(0, 1)

	for i, rate in enumerate(survival_rate['Survived']):
		plt.text(i, rate + 0.02, f'{rate:.2%}', ha='center')

	plt.tight_layout()
	plt.show()
	print("\nСтатистика выживших по полу:")
	print(f"Мужчины: {sex_counts['мужской']} пассажиров, Выжили: {survival_counts[1]['мужской']:.0f} ({survival_rate[survival_rate['Sex'] == 'мужской']['Survived'].iloc[0]:.2%})")
	print(f"Женщины: {sex_counts['женский']} пассажиров, Выжили: {survival_counts[1]['женский']:.0f} ({survival_rate[survival_rate['Sex'] == 'женский']['Survived'].iloc[0]:.2%})")

# Количество выживших по классу
def survival_by_class():
	class_survival = df.groupby(['Pclass', 'Survived']).size().unstack().fillna(0)
	class_counts = df['Pclass'].value_counts().sort_index()
	class_survival = class_survival.div(class_survival.sum(axis=1), axis=0) * 100
	plt.figure(figsize=(10, 6))
	class_survival.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff'])
	plt.title('Количество выживших по классу')
	plt.xlabel('Класс')
	plt.ylabel('Проценты пассажиров')
	plt.legend(['Погибли', 'Выжили'], title='Outcome')
	plt.tight_layout()
	plt.show()
	print("\nСтатистика выживших по классу:")
	for pclass in class_counts.index:
		survived_rate = class_survival[1].loc[pclass] / 100
		total = class_counts[pclass]
		survived = int(total * survived_rate)
		print(f"Класс {pclass}: {total} пассажиров, Выжили: {survived} ({survived_rate:.2%})")

# Выжившие по фамилиям
def survival_by_surname():
	surname_counts = df['Surname'].value_counts()

	top_surnames = surname_counts[surname_counts >= 3].index
	surname_data = df[df['Surname'].isin(top_surnames)].groupby('Surname').agg({
			'Survived': 'mean',
			'PassengerId': 'count',
			'Pclass': lambda x: x.mode()[0]
	}).reset_index()

	surname_data = surname_data.sort_values('Survived', ascending=False).head(10)
	plt.figure(figsize=(12, 6))
	sns.barplot(x='Survived', y='Surname', data=surname_data, color='#66b3ff')
	plt.title('Топ 10 выживших по кол-ву родственников (более 3 чел.)')
	plt.xlabel('Выжили')
	plt.ylabel('Фамилия')
	plt.xlim(0, 1)

	for i, rate in enumerate(surname_data['Survived']):
		plt.text(rate + 0.02, i, f'{rate:.2%}', va='center')

	plt.tight_layout()
	plt.show()
	print("\nСтатистика выживших по фамилии:")
	for _, row in surname_data.iterrows():
		print(f"Фамилия: {row['Surname']}, Пассажиры: {row['PassengerId']}, Выжили: {row['Survived']:.2%}, Класс: {row['Pclass']}")

survival_count()
survival_by_sex()
survival_by_class()
survival_by_surname()
