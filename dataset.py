import numpy as np
from PIL import Image, ImageDraw

class SatelliteDatasetGenerator:
    """
    Generates synthetic paired multi-sensor remote sensing data (Optical RGB, SAR, and Text)
    representing 5 distinct land cover classes. Also handles feature extraction.
    """
    def __init__(self, size=64):
        self.size = size
        self.classes = ['urban', 'forest', 'water', 'farmland', 'desert']
        self.vocabulary = [
            "urban", "building", "street", "concrete", "city",
            "forest", "tree", "vegetation", "wood", "river",
            "water", "lake", "canal", "agricultural", "field",
            "crop", "farmland", "desert", "sand", "dune"
        ]

    def generate_patch(self, class_name, has_river=False, has_road=False, seed=None):
        """
        Generates a single synthetic scene containing Optical and SAR images, and a Text description.
        """
        if seed is not None:
            np.random.seed(seed)
            
        opt_img = Image.new("RGB", (self.size, self.size))
        draw_opt = ImageDraw.Draw(opt_img)
        
        # Base clean backscatter grid for SAR sensor
        sar_clean = np.zeros((self.size, self.size), dtype=np.float32)
        
        # 1. Draw Land Cover Background Modalities
        if class_name == 'urban':
            # Base color: Grey concrete background
            draw_opt.rectangle([0, 0, self.size, self.size], fill=(175, 175, 175))
            sar_clean.fill(95.0) # Base urban surface backscatter
            
            # Buildings (rectangles)
            num_buildings = np.random.randint(4, 8)
            for _ in range(num_buildings):
                w = np.random.randint(8, 15)
                h = np.random.randint(8, 15)
                x = np.random.randint(2, self.size - w - 2)
                y = np.random.randint(2, self.size - h - 2)
                # Optical building color (reddish tiles/gray rooftops)
                building_color = (np.random.randint(120, 200), np.random.randint(100, 130), np.random.randint(90, 110))
                draw_opt.rectangle([x, y, x+w, y+h], fill=building_color, outline=(80, 80, 80))
                # SAR double-bounce: building core is moderately high backscatter, edges facing radar (top/left) are very bright
                sar_clean[y:y+h, x:x+w] = 190.0
                sar_clean[y:y+2, x:x+w] = 255.0 # Top edge reflection
                sar_clean[y:y+h, x:x+2] = 255.0 # Left edge reflection
                
            # Streets (asphalt roads)
            num_roads = np.random.randint(1, 3)
            for _ in range(num_roads):
                is_horizontal = np.random.choice([True, False])
                road_w = np.random.randint(3, 5)
                if is_horizontal:
                    y = np.random.randint(5, self.size - 5)
                    draw_opt.rectangle([0, y, self.size, y+road_w], fill=(70, 70, 70))
                    sar_clean[y:y+road_w, :] = 30.0 # Smooth asphalt roadway reflects radar away -> dark
                else:
                    x = np.random.randint(5, self.size - 5)
                    draw_opt.rectangle([x, 0, x+road_w, self.size], fill=(70, 70, 70))
                    sar_clean[:, x:x+road_w] = 30.0
                    
        elif class_name == 'forest':
            # Base color: Dark Forest Green
            draw_opt.rectangle([0, 0, self.size, self.size], fill=(25, 80, 30))
            sar_clean.fill(115.0) # Forest volume scattering
            
            # Tree crowns (many green overlapping circles)
            for _ in range(45):
                r = np.random.randint(3, 7)
                x = np.random.randint(0, self.size)
                y = np.random.randint(0, self.size)
                green_val = np.random.randint(90, 150)
                draw_opt.ellipse([x-r, y-r, x+r, y+r], fill=(20, green_val, 25))
                # Add local scattering variation for trees
                sar_clean[max(0, y-r):min(self.size, y+r), max(0, x-r):min(self.size, x+r)] += np.random.normal(5.0, 12.0)

        elif class_name == 'water':
            # Base color: Deep Ocean Blue/Teal
            draw_opt.rectangle([0, 0, self.size, self.size], fill=(20, 50, 100))
            sar_clean.fill(15.0) # Smooth water -> specular reflection -> very dark
            
            # Water ripples / currents
            for _ in range(6):
                y = np.random.randint(5, self.size-5)
                x1 = np.random.randint(0, self.size-20)
                x2 = x1 + np.random.randint(10, 30)
                draw_opt.line([x1, y, x2, y], fill=(25, 70, 120), width=1)
                
        elif class_name == 'farmland':
            # Base color: Bare Soil Brown
            draw_opt.rectangle([0, 0, self.size, self.size], fill=(130, 105, 80))
            sar_clean.fill(65.0)
            
            # Divide into 4 quadrants (agricultural plots)
            quads = [
                (0, 0, self.size//2, self.size//2),
                (self.size//2, 0, self.size, self.size//2),
                (0, self.size//2, self.size//2, self.size),
                (self.size//2, self.size//2, self.size, self.size)
            ]
            for q in quads:
                crop_type = np.random.choice(['young_green', 'ripe_yellow', 'dry_soil', 'ploughed'])
                if crop_type == 'young_green':
                    color = (np.random.randint(80, 110), np.random.randint(150, 190), np.random.randint(50, 80))
                    sar_val = np.random.randint(110, 140) # Dense crop canopy volume scattering
                elif crop_type == 'ripe_yellow':
                    color = (np.random.randint(180, 210), np.random.randint(160, 190), np.random.randint(60, 85))
                    sar_val = np.random.randint(95, 120)
                elif crop_type == 'dry_soil':
                    color = (np.random.randint(110, 130), np.random.randint(90, 110), np.random.randint(70, 90))
                    sar_val = np.random.randint(50, 65) # Smooth bare earth
                else: # ploughed
                    color = (np.random.randint(90, 110), np.random.randint(75, 90), np.random.randint(60, 75))
                    sar_val = np.random.randint(75, 95) # Rough surface backscatter
                    
                draw_opt.rectangle(q, fill=color, outline=(100, 85, 70))
                sar_clean[q[1]:q[3], q[0]:q[2]] = sar_val
                
                # Add crop rows (stripes)
                if np.random.choice([True, False]):
                    draw_lines = ImageDraw.Draw(opt_img)
                    for l in range(q[0]+2, q[2]-2, 4):
                        draw_lines.line([l, q[1]+2, l, q[3]-2], fill=(int(color[0]*0.85), int(color[1]*0.85), int(color[2]*0.85)), width=1)
                        
        elif class_name == 'desert':
            # Base color: Desert Sand Yellow
            draw_opt.rectangle([0, 0, self.size, self.size], fill=(225, 195, 135))
            sar_clean.fill(45.0) # Dry sand absorbs / reflects away radar
            
            # Draw dunes
            for _ in range(5):
                y_base = np.random.randint(5, self.size - 5)
                points = []
                for x in range(0, self.size + 10, 10):
                    y = y_base + int(4 * np.sin(x / 12.0))
                    points.append((x, y))
                draw_opt.line(points, fill=(205, 170, 115), width=2)
                # SAR backscatter edge highlights
                for pt in points:
                    px, py = pt
                    if 0 <= px < self.size and 0 <= py < self.size:
                        sar_clean[max(0, py-1):min(self.size, py+2), px:min(self.size, px+4)] = 75.0

        # 2. Add Overlay Modalities (Rivers / Highways)
        if has_river:
            # Winding river points
            river_points = []
            y = np.random.randint(15, self.size - 15)
            for x in range(0, self.size + 10, 8):
                y += np.random.randint(-4, 5)
                y = max(5, min(self.size - 5, y))
                river_points.append((x, y))
            # Optical blue water line
            draw_opt.line(river_points, fill=(22, 68, 135), width=5)
            # SAR water (specular / dark)
            for pt in river_points:
                px, py = pt
                if 0 <= px < self.size and 0 <= py < self.size:
                    sar_clean[max(0, py-3):min(self.size, py+4), max(0, px-3):min(self.size, px+4)] = 12.0
                    
        if has_road:
            # Straight highway
            is_horizontal = np.random.choice([True, False])
            road_w = 3
            if is_horizontal:
                y = np.random.randint(8, self.size - 8)
                draw_opt.line([0, y, self.size, y], fill=(65, 65, 65), width=road_w)
                sar_clean[max(0, y-1):min(self.size, y+2), :] = 25.0 # Smooth highway in SAR is dark
            else:
                x = np.random.randint(8, self.size - 8)
                draw_opt.line([x, 0, x, self.size], fill=(65, 65, 65), width=road_w)
                sar_clean[:, max(0, x-1):min(self.size, x+2)] = 25.0

        # 3. Simulate Multiplicative Speckle Noise for SAR
        speckle_std = 0.22
        shape = 1.0 / (speckle_std ** 2)
        scale = speckle_std ** 2
        speckle_noise = np.random.gamma(shape, scale, size=sar_clean.shape)
        
        # Multiplicative noise model
        sar_speckled = sar_clean * speckle_noise
        sar_speckled = np.clip(sar_speckled, 0, 255).astype(np.uint8)
        sar_img = Image.fromarray(sar_speckled, mode='L')
        
        # 4. Generate Text Descriptions
        description = self._generate_description(class_name, has_river, has_road)
        
        return {
            'optical': opt_img,
            'sar': sar_img,
            'description': description,
            'class': class_name,
            'has_river': has_river,
            'has_road': has_road
        }

    def _generate_description(self, class_name, has_river, has_road):
        templates = {
            'urban': [
                "An urban district with multiple buildings and concrete streets.",
                "A developed city block showing commercial buildings and road networks.",
                "A satellite view of a populated town showing buildings and transport corridors."
            ],
            'forest': [
                "A dense green forest canopy with lush vegetation and organic terrain.",
                "Thick forest patch displaying rich vegetation and high tree density.",
                "A natural woodland area covered with dense green trees."
            ],
            'water': [
                "A body of open water with a calm, deep blue surface.",
                "A natural lake or reservoir surrounded by surrounding terrain.",
                "A satellite view of a wide, dark water body."
            ],
            'farmland': [
                "A grid of agricultural farmland fields with cultivated crops.",
                "Rectangular crop plots with varying vegetation stages and agricultural boundaries.",
                "An aerial view of partitioned fields of crops and tilled soil."
            ],
            'desert': [
                "An arid desert landscape featuring wind-swept sand dunes.",
                "Dry sand dunes across a barren desert area.",
                "A vast expanse of sandy desert with wavy dune patterns."
            ]
        }
        
        desc = np.random.choice(templates[class_name])
        
        # Append modifiers to make features match across modalities
        if has_river and has_road:
            desc += " The landscape is intersected by a winding river and a main road."
        elif has_river:
            if class_name == 'urban':
                desc += " A canal passes directly through the urban area."
            elif class_name == 'water':
                pass
            else:
                desc += " A winding river cuts through the terrain."
        elif has_road:
            if class_name == 'urban':
                pass
            else:
                desc += " A straight roadway runs across the landscape."
                
        return desc

    def generate_dataset(self, num_samples=150, seed=42):
        """
        Generates a balanced multi-sensor remote sensing dataset.
        """
        np.random.seed(seed)
        dataset = []
        for i in range(num_samples):
            class_name = self.classes[i % len(self.classes)]
            # Add roads and rivers with reasonable probabilities
            has_river = (np.random.rand() < 0.25) and (class_name != 'water')
            has_road = (np.random.rand() < 0.25)
            
            patch = self.generate_patch(class_name, has_river, has_road)
            dataset.append(patch)
        return dataset

# --- Feature Extraction Helpers ---

def extract_optical_features(img):
    """
    Extracts a 17-dimensional feature vector from an Optical RGB image.
    Features: Mean R,G,B (3), Std R,G,B (3), R,G,B Color Histograms (9), Edge Density & Edge Mean (2)
    """
    arr = np.array(img, dtype=np.float32) / 255.0
    
    # 1. Color channel statistics
    r_mean, g_mean, b_mean = np.mean(arr[:, :, 0]), np.mean(arr[:, :, 1]), np.mean(arr[:, :, 2])
    r_std, g_std, b_std = np.std(arr[:, :, 0]), np.std(arr[:, :, 1]), np.std(arr[:, :, 2])
    
    # 2. Color histograms (3 bins per channel)
    hist_r, _ = np.histogram(arr[:, :, 0], bins=3, range=(0, 1))
    hist_g, _ = np.histogram(arr[:, :, 1], bins=3, range=(0, 1))
    hist_b, _ = np.histogram(arr[:, :, 2], bins=3, range=(0, 1))
    
    # L1 normalize histograms
    sum_r = np.sum(hist_r)
    sum_g = np.sum(hist_g)
    sum_b = np.sum(hist_b)
    hist_r = hist_r / (sum_r if sum_r > 0 else 1.0)
    hist_g = hist_g / (sum_g if sum_g > 0 else 1.0)
    hist_b = hist_b / (sum_b if sum_b > 0 else 1.0)
    
    # 3. Structural texture / Edge density
    # Convert to grayscale
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    # Standard differences for gradients aligned on a 63x63 grid
    dx = gray[:-1, 1:] - gray[:-1, :-1]
    dy = gray[1:, :-1] - gray[:-1, :-1]
    grad_mag = np.sqrt(dx**2 + dy**2)
    
    edge_density = np.mean(grad_mag > 0.05)
    edge_mean = np.mean(grad_mag)
    
    features = np.concatenate([
        [r_mean, g_mean, b_mean, r_std, g_std, b_std],
        hist_r, hist_g, hist_b,
        [edge_density, edge_mean]
    ])
    return features

def extract_sar_features(img):
    """
    Extracts a 6-dimensional feature vector from a SAR grayscale image.
    Features: Global mean (1), Global std (1), 4 Quadrant standard deviations for structural texture (4)
    """
    arr = np.array(img, dtype=np.float32) / 255.0
    
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    
    # Quadrant-based texture features (helps identify spatial distribution/structure)
    h, w = arr.shape
    q1 = arr[0:h//2, 0:w//2]
    q2 = arr[0:h//2, w//2:w]
    q3 = arr[h//2:h, 0:w//2]
    q4 = arr[h//2:h, w//2:w]
    q_stds = [np.std(q1), np.std(q2), np.std(q3), np.std(q4)]
    
    features = np.concatenate([
        [mean_val, std_val],
        q_stds
    ])
    return features

def extract_text_features(description, vocabulary):
    """
    Extracts keyword count vector for descriptions, normalized to unit length.
    """
    desc_lower = description.lower()
    features = np.zeros(len(vocabulary), dtype=np.float32)
    for idx, word in enumerate(vocabulary):
        count = desc_lower.count(word)
        features[idx] = float(count)
        
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm
        
    return features
