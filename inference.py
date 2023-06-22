from captum.attr import IntegratedGradients
import matplotlib
import numpy as np

def sample_example(dataset, filter_params=None):
    if not filter_params:  filter_params = {}
    ex_id = dataset.filter_load(**filter_params, format='index')[0]
    return ex_id

def Inference_IG(model, dataset, ex_id):
    ig = IntegratedGradients(model.forward_emb)
    example = dataset[ex_id][0].unsqueeze(dim=0)
    pred = model(example).argmax().cpu()
    
    # NOTE:  'baselines' should be zero tensor by default, which corresponds to having all <PAD> tokens.
    attributions = ig.attribute(inputs=model.get_embeddings(example), baselines=None, target=pred)
    scores = np.mean(attributions.detach().cpu().numpy(), axis=2).squeeze()

    encoding, label = dataset.__getitem__(ex_id, format='encoding')

    # create color mapping
    color_mapping = [   # RED => GREEN
        ((206, 35, 35), scores.min()),
        ((255, 255, 255), np.median(scores)),
        ((22, 206, 16), scores.max())
    ]
    
    # CREDIT:  https://databasecamp.de/en/ml/integrated-gradients-nlp  (reference for creating HTML display)
    def create_color_map(color_coords, color_bounds):
        def to_cmap_coord(x, level=0.0):  return( (level, np.interp(x, xp=[0,255], fp=[0,1]), np.interp(x, xp=[0,255], fp=[0,1])) )

        cmap_price_bounds = [np.interp(p, xp=[min(color_bounds), max(color_bounds)], fp=[0, 1]) for p in color_bounds]

        c_dict = {
            'red':tuple(to_cmap_coord(color_coords[i][0], cmap_price_bounds[i]) for i in range(len(color_coords))),
            'green':tuple(to_cmap_coord(color_coords[i][1], cmap_price_bounds[i]) for i in range(len(color_coords))),
            'blue':tuple(to_cmap_coord(color_coords[i][2], cmap_price_bounds[i]) for i in range(len(color_coords))),
        }
        
        return (matplotlib.colors.LinearSegmentedColormap('cmap', segmentdata=c_dict))
    c_map = create_color_map([c[0] for c in color_mapping], [c[1] for c in color_mapping])
    norm = matplotlib.colors.Normalize(vmin=scores.min(), vmax=scores.max())

    def build_html(text, c_map, norm, encoding, scores):
        def highlight(token, score):
            return f"<mark style=\"margin: 0; padding: 0; background-color:{matplotlib.colors.rgb2hex(c_map(norm(score)))}\">{token}</mark>"
        prev = (0, 0)
        cur_html = ""
        for i in range(len(encoding)):
            cur_html = cur_html + text[prev[1]: encoding.offsets[i][0]]
            cur_html = cur_html + highlight(encoding.tokens[i], scores[i])
            prev = encoding.offsets[i]
        return cur_html
    
    pred = pred.item()

    return {'html': build_html(dataset.__getitem__(ex_id, format='raw')[0], c_map, norm, encoding, scores), 'pred': pred+1, 'actual': label}