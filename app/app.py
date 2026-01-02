import streamlit as st
import pandas as pd
import numpy as np
try:
	import seaborn as sns
	sns_available = True
except Exception:
	sns = None
	sns_available = False
import matplotlib.pyplot as plt
try:
	import joblib
	joblib_available = True
except Exception:
	joblib = None
	joblib_available = False
from datetime import datetime

st.set_page_config(page_title='Simulación Ventas — Nov 2025', layout='wide')

PALETTE = {"primary": "#667eea", "accent": "#764ba2"}

# Styling para tarjetas y layout
APP_CSS = f"""
<style>
	.pv-card {{
		background: white;
		border-radius: 12px;
		box-shadow: 0 6px 18px rgba(102,126,234,0.08);
		padding: 22px;
		margin: 8px 6px;
	}}
	.pv-title {{
		font-size:14px; color:#6b7280; margin-bottom:6px;
	}}
	.pv-value {{
		font-size:28px; font-weight:700; color:#0f172a;
	}}
	.banner {{ background:#eef2ff; border-left:4px solid {PALETTE['primary']}; padding:12px 16px; border-radius:8px; color:#0b1220;}}
</style>
"""


@st.cache_data
def load_data(path='data/processed/inferencia_df_transformado.csv'):
	try:
		df = pd.read_csv(path, parse_dates=['fecha'])
		return df
	except Exception as e:
		st.error(f"No se pudo cargar el archivo de inferencia: {e}")
		return None


@st.cache_resource
def load_model(path='models/modelo_final.joblib'):
	if not joblib_available:
		st.error('La librería `joblib` no está instalada en el entorno. Instala `joblib` y reinicia la app.')
		return None
	try:
		model = joblib.load(path)
		return model
	except Exception as e:
		st.error(f"No se pudo cargar el modelo: {e}")
		return None


def format_eur(x):
	return f"€{x:,.2f}"


def simulate_recursive(df_prod, model, feature_cols):
	df = df_prod.sort_values('fecha').reset_index(drop=True).copy()

	# Initialize rolling lags from the first row (these were computed from octubre)
	required_lags = [f'unidades_vendidas_lag_{i}' for i in range(1, 8)]
	for c in required_lags + ['unidades_vendidas_ma7']:
		if c not in df.columns:
			raise KeyError(f'Falta la columna requerida: {c}')

	# last_7: most recent first -> [lag1, lag2, ..., lag7]
	last_7 = [float(df.at[0, f]) for f in required_lags]

	preds = []
	ingresos = []

	for i in range(len(df)):
		row = df.loc[i].copy()

		# For day 1 use the lags as they are in the file (already in row)
		if i == 0:
			for j, lag_col in enumerate(required_lags):
				row[lag_col] = df.at[0, lag_col]
			row['unidades_vendidas_ma7'] = df.at[0, 'unidades_vendidas_ma7']
		else:
			# update lags from last_7
			for j, lag_col in enumerate(required_lags):
				row[lag_col] = last_7[j]
			# update ma7 as mean of last 7 values
			row['unidades_vendidas_ma7'] = float(np.mean(last_7))

		# Build feature vector in correct order
		X_row = row.reindex(feature_cols).to_frame().T.fillna(0)

		# Predict
		pred = model.predict(X_row)[0]
		if np.isnan(pred):
			pred = 0.0
		preds.append(float(pred))

		# update ingresos: price already adjusted in df
		ingreso = float(pred) * float(row.get('precio_venta', 0.0))
		ingresos.append(ingreso)

		# roll the last_7: insert pred at front, drop last
		last_7 = [float(pred)] + last_7[:6]

	df['unidades_pred'] = np.round(preds).astype(int)
	df['ingresos_pred'] = np.round(ingresos, 2)
	return df


def fill_lags_from_history(df_inf, ventas_path='data/raw/entrenamiento/ventas.csv'):
	"""Si faltan columnas de lag en df_inf, intentar reconstruirlas desde el histórico de ventas.
	Añade columnas: unidades_vendidas_lag_1..7 y unidades_vendidas_ma7 cuando faltan.
	"""
	required_lags = [f'unidades_vendidas_lag_{i}' for i in range(1, 8)] + ['unidades_vendidas_ma7']
	missing = [c for c in required_lags if c not in df_inf.columns]
	if not missing:
		return df_inf, False

	try:
		hist = pd.read_csv(ventas_path, parse_dates=['fecha'])
	except Exception as e:
		st.warning(f'No se encontraron lags y no se pudo leer el histórico: {e}')
		return df_inf, False

	# ensure columnas
	if 'producto_id' not in df_inf.columns or 'producto_id' not in hist.columns or 'unidades_vendidas' not in hist.columns:
		st.warning('No es posible reconstruir lags: faltan columnas `producto_id` o `unidades_vendidas` en los datos.')
		return df_inf, False

	hist = hist.sort_values(['producto_id', 'fecha'])

	# compute last 7 ventas for each product prior to the first inference date
	added = False
	for pid in df_inf['producto_id'].unique():
		prod_inf = df_inf[df_inf['producto_id'] == pid]
		if prod_inf.empty:
			continue
		first_inf_date = prod_inf['fecha'].min()
		hist_prod = hist[(hist['producto_id'] == pid) & (hist['fecha'] < first_inf_date)].sort_values('fecha')
		last_vals = hist_prod['unidades_vendidas'].dropna().astype(float).tolist()
		# take most recent first
		last_vals = last_vals[-7:][::-1] if last_vals else []
		# pad to length 7
		if len(last_vals) < 7:
			pad = [0.0] * (7 - len(last_vals))
			last_vals = last_vals + pad

		# ma7 as mean of the 7 values
		ma7 = float(np.mean(last_vals)) if last_vals else 0.0

		# set these values on the first row of df_inf for this product
		mask = (df_inf['producto_id'] == pid)
		for i, val in enumerate(last_vals, start=1):
			df_inf.loc[mask, f'unidades_vendidas_lag_{i}'] = val
		df_inf.loc[mask, 'unidades_vendidas_ma7'] = ma7
		added = True

	if added:
		st.info('Se han generado lags iniciales desde el histórico de ventas para los productos faltantes.')
	return df_inf, added


def run_simulation_for_product(df_all, model, producto_nombre, discount_pct, comp_multiplier):
	df_prod = df_all[df_all['nombre'] == producto_nombre].copy()
	if df_prod.empty:
		st.error('No hay datos para el producto seleccionado.')
		return None

	# Apply discount to precio_venta (same for all days)
	df_prod['precio_venta'] = df_prod['precio_base'] * (1 - discount_pct / 100.0)

	# Adjust competitor platform prices
	comp_cols = ['Amazon', 'Decathlon', 'Deporvillage']
	for c in comp_cols:
		if c in df_prod.columns:
			df_prod[c] = df_prod[c] * comp_multiplier

	# Recalculate precio_competencia as the minimum competitor price available
	if set(comp_cols).issubset(df_prod.columns):
		df_prod['precio_competencia'] = df_prod[comp_cols].min(axis=1)

	# Recalculate discount_porcentaje and ratio_precio
	df_prod['descuento_porcentaje'] = ((df_prod['precio_base'] - df_prod['precio_venta']) / df_prod['precio_base']) * 100
	# Avoid division by zero
	df_prod['ratio_precio'] = df_prod['precio_venta'] / df_prod['precio_competencia'].replace({0: np.nan})

	# Determine feature columns expected by the model
	try:
		feature_cols = list(model.feature_names_in_)
	except Exception:
		feature_cols = [c for c in df_prod.columns if c != 'unidades_vend']

	# Ensure feature columns exist; if not, fill missing with zeros
	for c in feature_cols:
		if c not in df_prod.columns:
			df_prod[c] = 0

	# Run recursive predictions with spinner and progress
	with st.spinner('Calculando predicciones recursivas día a día...'):
		result_df = simulate_recursive(df_prod, model, feature_cols)

	return result_df


def main():
	st.markdown("<h1 style='color:{}'>📈 Simulación Ventas — Noviembre 2025</h1>".format(PALETTE['primary']), unsafe_allow_html=True)

	df = load_data()
	model = load_model()

	if df is None or model is None:
		return

	# Si faltan lags en el CSV de inferencia, intentar reconstruirlos desde ventas históricas
	df, lags_added = fill_lags_from_history(df)

	# Sidebar controls
	st.sidebar.header('Controles de Simulación')

	productos = df['nombre'].unique().tolist()
	producto = st.sidebar.selectbox('Producto', productos)

	descuento = st.sidebar.slider('Ajuste de descuento (%)', -50, 50, 0, step=5)

	escenario = st.sidebar.radio('Escenario de competencia', ['Actual (0%)', 'Competencia -5%', 'Competencia +5%'])
	escenario_map = {'Actual (0%)': 1.0, 'Competencia -5%': 0.95, 'Competencia +5%': 1.05}
	comp_mult = escenario_map[escenario]

	sim_btn = st.sidebar.button('Simular Ventas')

	st.sidebar.markdown('---')
	st.sidebar.info('Pulsa "Simular Ventas" para ejecutar la predicción recursiva.')

	# Render preview cards antes de simular
	st.markdown(APP_CSS, unsafe_allow_html=True)
	st.markdown(f"<h2 style='color:{PALETTE['primary']}'>📊 Dashboard de Predicción - Noviembre 2025</h2>", unsafe_allow_html=True)

	# Mostrar información del producto seleccionado en tarjetas
	prod_row = df[df['nombre'] == producto]
	if not prod_row.empty:
		prod_row = prod_row.iloc[0]
		cat = prod_row.get('categoria', '—')
		subcat = prod_row.get('subcategoria', '—')
		precio_base = prod_row.get('precio_base', 0.0)
	else:
		cat, subcat, precio_base = '—', '—', 0.0

	st.markdown('<div class="banner">👉 Configura los parámetros en el panel lateral y presiona "Simular Ventas" para ver las predicciones</div>', unsafe_allow_html=True)
	c1, c2, c3 = st.columns([1,1,1])
	c1.markdown(f"<div class='pv-card'><div class='pv-title'>📦 Categoría</div><div class='pv-value'>{cat}</div></div>", unsafe_allow_html=True)
	c2.markdown(f"<div class='pv-card'><div class='pv-title'>🔖 Subcategoría</div><div class='pv-value'>{subcat}</div></div>", unsafe_allow_html=True)
	c3.markdown(f"<div class='pv-card'><div class='pv-title'>💶 Precio Base</div><div class='pv-value'>{format_eur(precio_base)}</div></div>", unsafe_allow_html=True)

	st.markdown('---')

	if not sim_btn:
		st.markdown('<div style="font-size:2rem; font-weight:700; color:#0f172a; margin-bottom:0.5em;">Vista Previa</div>', unsafe_allow_html=True)
		st.write("Presiona 'Simular Ventas' para ver las predicciones reales y los KPIs.")

	if sim_btn:
		st.markdown('<div style="font-size:2rem; font-weight:700; color:#0f172a; margin-bottom:0.5em;">Predicción diaria de ventas</div>', unsafe_allow_html=True)
		# Run main simulation for selected product
		try:
			df_result = run_simulation_for_product(df, model, producto, descuento, comp_mult)
			if df_result is None:
				return
		except KeyError as e:
			st.error(f'Falta columna requerida: {e}')
			return
		except Exception as e:
			st.error(f'Error durante la simulación: {e}')
			return

		# KPIs
		total_unidades = int(df_result['unidades_pred'].sum())
		total_ingresos = float(df_result['ingresos_pred'].sum())
		precio_promedio = float(df_result['precio_venta'].mean())
		descuento_promedio = float(df_result['descuento_porcentaje'].mean())

		kpi_cols = st.columns(4)
		kpi_cols[0].metric('Unidades Totales', f"{total_unidades:,d}")
		kpi_cols[1].metric('Ingresos Proyectados', format_eur(total_ingresos))
		kpi_cols[2].metric('Precio Promedio', format_eur(precio_promedio))
		kpi_cols[3].metric('Descuento Promedio', f"{descuento_promedio:.1f}%")

		st.markdown('---')

		# Daily prediction plot
		fig, ax = plt.subplots(figsize=(10, 4))
		# Use seaborn if available, else fallback to matplotlib
		if sns_available:
			sns.lineplot(data=df_result, x=df_result['fecha'].dt.day, y='unidades_pred', color=PALETTE['primary'], marker='o', ax=ax)
		else:
			st.warning('La librería `seaborn` no está instalada — usando matplotlib como fallback. Para mejor apariencia instala `seaborn`.')
			ax.plot(df_result['fecha'].dt.day, df_result['unidades_pred'], color=PALETTE['primary'], marker='o')
		ax.set_xlabel('Día de noviembre')
		ax.set_ylabel('Unidades vendidas (pred)')
		ax.set_title(f'Predicción diaria — {producto}')

		# Mark Black Friday (28 Nov)
		bf_day = 28
		if any(df_result['fecha'].dt.day == bf_day):
			bf_val = df_result.loc[df_result['fecha'].dt.day == bf_day, 'unidades_pred'].values[0]
			ax.axvline(bf_day, color='red', linestyle='--', alpha=0.6)
			ax.scatter([bf_day], [bf_val], color='red', zorder=5)
			ax.annotate('Black Friday 🎯', xy=(bf_day, bf_val), xytext=(bf_day+1, bf_val*1.05), color='red')

		st.pyplot(fig)

		st.markdown('---')

		# Table with details
		table = df_result[['fecha', 'dia_semana', 'precio_venta', 'precio_competencia', 'descuento_porcentaje', 'unidades_pred', 'ingresos_pred']].copy()
		table['fecha'] = table['fecha'].dt.date
		table.rename(columns={'dia_semana': 'día_semana', 'precio_venta': 'precio_venta (€)', 'precio_competencia': 'precio_competencia (€)', 'descuento_porcentaje': 'descuento (%)', 'unidades_pred': 'unidades', 'ingresos_pred': 'ingresos (€)'}, inplace=True)

		def highlight_bf(row):
			day = pd.to_datetime(row['fecha']).day
			if day == bf_day:
				return ['background-color: #fff2f0'] * len(row)
			return [''] * len(row)

		styled = table.style.apply(highlight_bf, axis=1).format({ 'precio_venta (€)': '{:,.2f}', 'precio_competencia (€)': '{:,.2f}', 'descuento (%)': '{:.1f}', 'unidades': '{:,.0f}', 'ingresos (€)': '€{:,.2f}' })
		st.dataframe(styled, use_container_width=True)

		st.markdown('---')

		# Comparative scenarios: compute totals for -5%, 0%, +5%
		scenarios = {'Competencia -5%': 0.95, 'Actual (0%)': 1.0, 'Competencia +5%': 1.05}
		comp_results = {}
		with st.spinner('Calculando comparativa de escenarios...'):
			for name, mult in scenarios.items():
				r = run_simulation_for_product(df, model, producto, descuento, mult)
				comp_results[name] = {'unidades': int(r['unidades_pred'].sum()), 'ingresos': float(r['ingresos_pred'].sum())}

		st.subheader('Comparativa de escenarios de competencia')
		cols = st.columns(3)
		i = 0
		for name, vals in comp_results.items():
			cols[i].metric(name, f"{vals['unidades']:,d}", delta=format_eur(vals['ingresos']))
			i += 1

		st.success('Simulación completada ✅')


if __name__ == '__main__':
	main()

