try:
    import cairosvg
except ImportError:
    CAIROSVG_INSTALLED = False
else:
    CAIROSVG_INSTALLED = True

def display_svg_with_zoom(graph, width_zoom_percent=100, height_zoom_percent=100):
    """
    Display a graphviz object in Jupyter with specified zoom percents for width and height, using PNG conversion.
    
    Parameters:
    - graph: A graphviz.Source object.
    - width_zoom_percent: The zoom percentage for width. 100 means no zoom.
    - height_zoom_percent: The zoom percentage for height. 100 means no zoom.
    """
    if not CAIROSVG_INSTALLED:
        raise ImportError("cairosvg is required for this functionality. Please install it using 'pip install cairosvg' or 'pip install modelrunner[svg]'")

    # Convert the graphviz Source object to an SVG string
    svg_str = graph.pipe(format='svg').decode('utf-8')
    
    # Use cairosvg to convert SVG string to PNG
    output = BytesIO()
    cairosvg.svg2png(bytestring=svg_str.encode('utf-8'), write_to=output)
    
    # Load the PNG image with PIL and display with desired zoom
    img = Image.open(output)
    width, height = img.size
    new_width = int(width * (width_zoom_percent / 100))
    new_height = int(height * (height_zoom_percent / 100))
    img_resized = img.resize((new_width, new_height))
    
    display(img_resized)