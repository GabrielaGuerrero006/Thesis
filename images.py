import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from database import (
    get_num_exportables_no_exportables, get_num_verdes_maduros, get_num_con_defectos_sin_defectos,
    get_confianza_promedio_exportabilidad, get_confianza_promedio_madurez, get_confianza_promedio_defectos
)

def ensure_lote_dir_exists(lote_number):
    """
    Verifica si existe la carpeta images/<lote_number>, si no existe la crea.
    """
    dir_path = os.path.join('images', str(lote_number))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def generar_grafico_exportables_pie(lote_number):
    """
    Genera un gráfico de pastel con los porcentajes de mangos exportables y no exportables para un lote dado.
    Guarda la imagen en images/<lote_number>/Exportables-NoExportables-Pie.jpg
    Args:
        lote_number (str o int): Número de lote
    Returns:
        str: Ruta al archivo de imagen generado
    """
    # Obtener datos desde la base de datos
    datos = get_num_exportables_no_exportables(lote_number)
    exportables = datos.get('exportable', 0)
    no_exportables = datos.get('no_exportable', 0)
    total = exportables + no_exportables
    if total == 0:
        # Evitar división por cero, crear gráfico vacío
        sizes = [1]
        labels = ['Sin datos']
        colors = ['#cccccc']
    else:
        sizes = [exportables, no_exportables]
        labels = ['Exportables', 'No Exportables']
        colors = ['#4CAF50', '#F44336']

    # Crear gráfico de pastel
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.set_title('Porcentajes de Mangos Exportables/No Exportables')
    ax.axis('equal')  # Para que sea un círculo

    # Guardar imagen en la ruta correspondiente
    dir_path = ensure_lote_dir_exists(lote_number)
    img_path = os.path.join(dir_path, 'Exportables-NoExportables-Pie.jpg')
    plt.savefig(img_path, bbox_inches='tight')
    plt.close(fig)
    return img_path

def generar_grafico_verdes_maduros_pie(lote_number):
    """
    Genera un gráfico de pastel con los porcentajes de mangos verdes y maduros para un lote dado.
    Guarda la imagen en images/<lote_number>/Verdes-Maduros-Pie.jpg
    Args:
        lote_number (str o int): Número de lote
    Returns:
        str: Ruta al archivo de imagen generado
    """
    datos = get_num_verdes_maduros(lote_number)
    verdes = datos.get('mango_verde', 0)
    maduros = datos.get('mango_maduro', 0)
    total = verdes + maduros
    if total == 0:
        sizes = [1]
        labels = ['Sin datos']
        colors = ['#cccccc']
    else:
        sizes = [verdes, maduros]
        labels = ['Verdes', 'Maduros']
        colors = ['#2196F3', '#FFEB3B']  # Azul y amarillo

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.set_title('Porcentajes de Mangos Verde/Maduros')
    ax.axis('equal')

    dir_path = ensure_lote_dir_exists(lote_number)
    img_path = os.path.join(dir_path, 'Verdes-Maduros-Pie.jpg')
    plt.savefig(img_path, bbox_inches='tight')
    plt.close(fig)
    return img_path

def generar_grafico_con_sin_defectos_pie(lote_number):
    """
    Genera un gráfico de pastel con los porcentajes de mangos con y sin defectos para un lote dado.
    Guarda la imagen en images/<lote_number>/Con-Sin-Defectos-Pie.jpg
    Args:
        lote_number (str o int): Número de lote
    Returns:
        str: Ruta al archivo de imagen generado
    """
    datos = get_num_con_defectos_sin_defectos(lote_number)
    con_defectos = datos.get('mango_con_defectos', 0)
    sin_defectos = datos.get('mango_sin_defectos', 0)
    total = con_defectos + sin_defectos
    if total == 0:
        sizes = [1]
        labels = ['Sin datos']
        colors = ['#cccccc']
    else:
        sizes = [con_defectos, sin_defectos]
        labels = ['Con Defectos', 'Sin Defectos']
        colors = ['#9C27B0', '#8BC34A']  # Morado y verde claro

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.set_title('Porcentajes de Mangos Sin/Con Defectos')
    ax.axis('equal')

    dir_path = ensure_lote_dir_exists(lote_number)
    img_path = os.path.join(dir_path, 'Con-Sin-Defectos-Pie.jpg')
    plt.savefig(img_path, bbox_inches='tight')
    plt.close(fig)
    return img_path

def generar_grafico_confianza_promedio_bar(lote_number):
    """
    Genera un gráfico de barras con los porcentajes de confianza promedio de exportabilidad, madurez y defectos para un lote dado.
    Guarda la imagen en images/<lote_number>/Confianza-Promedio-Bar.jpg
    Args:
        lote_number (str o int): Número de lote
    Returns:
        str: Ruta al archivo de imagen generado
    """
    confianza_exportabilidad = get_confianza_promedio_exportabilidad(lote_number)
    confianza_madurez = get_confianza_promedio_madurez(lote_number)
    confianza_defectos = get_confianza_promedio_defectos(lote_number)

    categorias = ['Exportabilidad', 'Madurez', 'Defectos']
    valores = [confianza_exportabilidad, confianza_madurez, confianza_defectos]
    colores = ['#4CAF50', '#FFEB3B', '#9C27B0']  # Verde, amarillo, morado

    fig, ax = plt.subplots()
    bars = ax.bar(categorias, valores, color=colores)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Porcentaje de Confianza Promedio (%)')
    ax.set_title('Porcentajes de Confianza Promedio')
    for bar, valor in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f'{valor:.2f}%', ha='center', va='bottom')

    dir_path = ensure_lote_dir_exists(lote_number)
    img_path = os.path.join(dir_path, 'Confianza-Promedio-Bar.jpg')
    plt.savefig(img_path, bbox_inches='tight')
    plt.close(fig)
    return img_path