"""
Regenerates the three worked-example tree figures (forest.pdf, forest2.pdf, forest3.pdf)
programmatically, matching the original visual style, but with forest2 (the mirrored tree) FIXED:
the original hand-made figure has the -70k and -60k threshold labels swapped between branches
relative to what correct mirroring actually produces (confirmed by re-deriving the mirrored tree
by hand from forest3's structure, and cross-checked against two concrete predictions -- see
reviews/findings.md).

Saved as forest_v2.pdf, forest2_v2.pdf, forest3_v2.pdf (new files, originals untouched) for use in
paper_v2.tex.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import FancyArrowPatch

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'


def draw_tree(nodes, edges, out_path, figsize=(9.5, 5)):
    """
    nodes: dict[name] -> (x, y, label)
    edges: list of (parent, child, edge_label, label_offset_x)
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.3, 10.8)
    ax.set_ylim(-0.1, 6)
    ax.axis('off')

    boxes = {}
    for name, (x, y, label) in nodes.items():
        is_leaf = not any(p == name for p, _, _, _ in edges)
        w = 0.09 * len(label) + 0.35
        h = 0.55
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                              boxstyle="round,pad=0.08,rounding_size=0.12",
                              linewidth=1.1, edgecolor='black', facecolor='white', zorder=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, zorder=3)
        boxes[name] = (x, y, w, h)

    for parent, child, edge_label, label_dx in edges:
        px, py, pw, ph = boxes[parent]
        cx, cy, cw, ch = boxes[child]
        start = (px + (0.28 if cx > px else (-0.28 if cx < px else 0)), py - ph / 2)
        end = (cx, cy + ch / 2)
        arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=14,
                                 linewidth=1.0, color='black', zorder=1,
                                 connectionstyle="arc3,rad=0.0")
        ax.add_patch(arrow)
        mx, my = (start[0] + end[0]) / 2 + label_dx, (start[1] + end[1]) / 2
        ax.text(mx, my, edge_label, ha='center', va='center', fontsize=11,
                style='italic', backgroundcolor='white', zorder=4)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def tree1(op_symbol, out_path):
    """forest.pdf (op='\\leq') / forest3.pdf (op='<') -- the two original operator variants."""
    nodes = {
        'root': (5.0, 5.0, f'self-employed ${op_symbol}$ 0.5'),
        'income70': (2.5, 3.4, f'income ${op_symbol}$ 70k'),
        'defaults1': (1.2, 1.8, 'defaults'),
        'pays1': (3.8, 1.8, 'pays'),
        'dependents': (7.5, 3.4, f'dependents ${op_symbol}$ 3'),
        'pays2': (6.3, 1.8, 'pays'),
        'income60': (8.7, 1.8, f'income ${op_symbol}$ 60k'),
        'defaults2': (7.7, 0.4, 'defaults'),
        'pays3': (9.7, 0.4, 'pays'),
    }
    edges = [
        ('root', 'income70', 'yes', -0.15),
        ('root', 'dependents', 'no', 0.15),
        ('income70', 'defaults1', 'yes', -0.1),
        ('income70', 'pays1', 'no', 0.1),
        ('dependents', 'pays2', 'yes', -0.1),
        ('dependents', 'income60', 'no', 0.1),
        ('income60', 'defaults2', 'yes', -0.1),
        ('income60', 'pays3', 'no', 0.1),
    ]
    draw_tree(nodes, edges, out_path)


def tree2_corrected(out_path):
    """
    forest2.pdf, CORRECTED. Mirroring forest3 (operator '<'): recursively swap left/right children
    and negate thresholds at every node, applied to the ENTIRE tree from the root. Re-derived by
    hand and cross-checked against two concrete predictions (see reviews/findings.md) -- the
    original hand-made figure had the -70k/-60k labels swapped relative to this.
    """
    nodes = {
        'root': (5.0, 5.0, r'-self-employed $\leq$ -0.5'),
        'dependents': (2.8, 3.4, r'-dependents $\leq$ -3'),
        'income60': (1.6, 1.8, r'-income $\leq$ -60k'),
        'pays_a': (0.6, 0.4, 'pays'),
        'defaults_a': (2.6, 0.4, 'defaults'),
        'pays_b': (4.2, 1.8, 'pays'),
        'income70': (7.6, 3.4, r'-income $\leq$ -70k'),
        'pays_c': (6.6, 1.8, 'pays'),
        'defaults_b': (8.6, 1.8, 'defaults'),
    }
    edges = [
        ('root', 'dependents', 'yes', -0.15),
        ('root', 'income70', 'no', 0.15),
        ('dependents', 'income60', 'yes', -0.1),
        ('dependents', 'pays_b', 'no', 0.1),
        ('income60', 'pays_a', 'yes', -0.1),
        ('income60', 'defaults_a', 'no', 0.1),
        ('income70', 'pays_c', 'yes', -0.1),
        ('income70', 'defaults_b', 'no', 0.1),
    ]
    draw_tree(nodes, edges, out_path)


if __name__ == '__main__':
    out_dir = '/home/gykovacs/workspaces/good/conditioning_bias/paper_2023_conditioning_bias_springer'
    tree1(r'\leq', f'{out_dir}/forest_v2.pdf')
    tree1('<', f'{out_dir}/forest3_v2.pdf')
    tree2_corrected(f'{out_dir}/forest2_v2.pdf')
    print("done")
