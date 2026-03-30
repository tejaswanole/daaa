import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def draw_box(ax, xy, width, height, text, title=None, facecolor='#e3f2fd', edgecolor='#1565c0'):
    box = mpatches.FancyBboxPatch(xy, width, height,
            boxstyle="round,pad=0.1", facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
    ax.add_patch(box)
    
    if title:
        # Title
        ax.text(xy[0] + width/2, xy[1] + height - 0.25, title,
                ha='center', va='center', fontsize=12, fontweight='bold', color=edgecolor)
        # Content
        ax.text(xy[0] + width/2, xy[1] + height/2 - 0.15, text,
                ha='center', va='center', fontsize=10, color='#333333', multialignment='center')
    else:
        ax.text(xy[0] + width/2, xy[1] + height/2, text,
                ha='center', va='center', fontsize=10.5, color='#333333', fontweight='bold', multialignment='center')

def draw_diamond(ax, xy, width, height, text, facecolor='#fff3e0', edgecolor='#e65100'):
    cx = xy[0] + width/2
    cy = xy[1] + height/2
    points = [
        [cx, xy[1] + height],
        [xy[0] + width, cy],
        [cx, xy[1]],
        [xy[0], cy]
    ]
    diamond = plt.Polygon(points, facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
    ax.add_patch(diamond)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=11, color='#bf360c', fontweight='bold', multialignment='center')

def draw_arrow(ax, start, end, connectionstyle="arc3"):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#555', connectionstyle=connectionstyle))

def add_label(ax, start, end, text, text_offset=(0,0)):
    mx = (start[0] + end[0]) / 2 + text_offset[0]
    my = (start[1] + end[1]) / 2 + text_offset[1]
    ax.text(mx, my, text, ha='center', va='center', fontsize=9.5, color='#555', backgroundcolor='white', style='italic', zorder=10)

def draw_line(ax, start, end):
    ax.plot([start[0], end[0]], [start[1], end[1]], color='#555', lw=1.5, zorder=1)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(base_dir, 'AIRDA_Figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(24, 9))
    ax.set_xlim(0, 29)
    ax.set_ylim(0, 10.5)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    step_fc = '#e3f2fd'
    step_ec = '#1565c0'
    action_fc = '#e8f5e9'
    action_ec = '#2e7d32'
    dec_fc = '#fff3e0'
    dec_ec = '#e65100'
    anom_fc = '#ffebee'
    anom_ec = '#c62828'
    
    ax.text(14.5, 9.8, 'AIRDA Framework: End-to-End Workflow Pipeline (Horizontal)',
            ha='center', va='center', fontsize=20, fontweight='bold', color='#1a1a2e')
    
    w, h = 3.0, 1.5
    
    y_center = 4.0
    y_top = 7.5
    y_bottom = 0.5
    
    M_xy = (0.5, y_center)
    D_xy = (4.0, y_center - 0.25)
    D_w, D_h = 3.6, h + 0.5
    Crit_xy = (4.3, y_top)
    
    P_xy = (8.2, y_center)
    C_xy = (11.8, y_center)
    A_xy = (15.4, y_center)
    
    S_xy = (19.0, y_center - 0.25)
    S_w, S_h = 3.6, h + 0.5
    SU_xy = (19.3, y_top)
    SD_xy = (19.3, y_bottom)
    
    V_xy = (23.2, y_center)
    SLA_xy = (26.6, y_center - 0.25)
    SLA_w, SLA_h = 2.2, D_h
    Log_xy = (23.2, y_top)
    
    draw_box(ax, M_xy, w, h, 'Collect 9D Resource Metrics (x_t)', title='Step 1: MONITOR', facecolor=step_fc, edgecolor=step_ec)
    draw_diamond(ax, D_xy, D_w, D_h, 'Step 2: DETECT\nAnomaly\nScore > \u03B1?', facecolor=dec_fc, edgecolor=dec_ec)
    draw_box(ax, Crit_xy, w-0.6, h, 'Flag for Priority\nEscalate to Critical', facecolor=anom_fc, edgecolor=anom_ec)
    
    draw_box(ax, P_xy, w, h, 'K-Means Workload Clustering\nAssign cluster c_j', title='Step 3: PROFILE', facecolor=step_fc, edgecolor=step_ec)
    draw_box(ax, C_xy, w, h, "GA-RF Classifier Predicts Tier\ny\u0302_t with x'_t = [x_t, c_j]", title='Step 4: CLASSIFY', facecolor=step_fc, edgecolor=step_ec)
    draw_box(ax, A_xy, w, h, 'Map y\u0302_t to Resource Policy\n(Low/Med/High/Critical)', title='Step 5: ALLOCATE', facecolor=action_fc, edgecolor=action_ec)
    
    draw_diamond(ax, S_xy, S_w, S_h, 'Step 6: SCALE\nScaling Conditions\nMet?', facecolor=dec_fc, edgecolor=dec_ec)
    draw_box(ax, SU_xy, w-0.6, h, 'Trigger Pre-emptive\nScale-Up', facecolor=action_fc, edgecolor=action_ec)
    draw_box(ax, SD_xy, w-0.6, h, 'Trigger\nScale-Down', facecolor=action_fc, edgecolor=action_ec)
    
    draw_box(ax, V_xy, w, h, 'Check Response & Uptime SLA', title='Step 7: VALIDATE', facecolor=step_fc, edgecolor=step_ec)
    draw_diamond(ax, SLA_xy, SLA_w, SLA_h, 'SLA\nViolated?', facecolor=dec_fc, edgecolor=dec_ec)
    draw_box(ax, Log_xy, w, h, 'Log Violation &\nAdjust Tier Upward', facecolor=anom_fc, edgecolor=anom_ec)
    
    cy = y_center + h/2
    draw_arrow(ax, (M_xy[0] + w, cy), (D_xy[0], cy))
    
    draw_arrow(ax, (D_xy[0] + D_w, cy), (P_xy[0], cy))
    add_label(ax, (D_xy[0] + D_w, cy), (P_xy[0], cy), 'No')
    
    draw_arrow(ax, (D_xy[0] + D_w/2, D_xy[1] + D_h), (Crit_xy[0] + (w-0.6)/2, Crit_xy[1]))
    add_label(ax, (D_xy[0] + D_w/2, D_xy[1] + D_h), (Crit_xy[0] + (w-0.6)/2, Crit_xy[1]), 'Yes')
    
    # Anomaly -> Allocate
    draw_line(ax, (Crit_xy[0] + w-0.6, Crit_xy[1] + h/2), (A_xy[0] + w/2 - 0.2, Crit_xy[1] + h/2))
    draw_arrow(ax, (A_xy[0] + w/2 - 0.2, Crit_xy[1] + h/2), (A_xy[0] + w/2 - 0.2, A_xy[1] + h))
    
    draw_arrow(ax, (P_xy[0] + w, cy), (C_xy[0], cy))
    draw_arrow(ax, (C_xy[0] + w, cy), (A_xy[0], cy))
    draw_arrow(ax, (A_xy[0] + w, cy), (S_xy[0], cy))
    
    # Scale -> ScaleUp
    draw_arrow(ax, (S_xy[0] + S_w/2, S_xy[1] + S_h), (SU_xy[0] + (w-0.6)/2, SU_xy[1]))
    add_label(ax, (S_xy[0] + S_w/2, S_xy[1] + S_h), (SU_xy[0] + (w-0.6)/2, SU_xy[1]), 'Upward Trend')
    
    # Scale -> ScaleDown
    draw_arrow(ax, (S_xy[0] + S_w/2, S_xy[1]), (SD_xy[0] + (w-0.6)/2, SD_xy[1] + h))
    add_label(ax, (S_xy[0] + S_w/2, S_xy[1]), (SD_xy[0] + (w-0.6)/2, SD_xy[1] + h), 'Util < 20%')
    
    # Scale -> Validate
    draw_arrow(ax, (S_xy[0] + S_w, cy), (V_xy[0], cy))
    add_label(ax, (S_xy[0] + S_w, cy), (V_xy[0], cy), 'Stable')
    
    # ScaleUp -> Validate
    draw_line(ax, (SU_xy[0] + w-0.6, SU_xy[1] + h/2), (V_xy[0] + w/2 - 0.2, SU_xy[1] + h/2))
    draw_arrow(ax, (V_xy[0] + w/2 - 0.2, SU_xy[1] + h/2), (V_xy[0] + w/2 - 0.2, V_xy[1] + h))
    
    # ScaleDown -> Validate
    draw_line(ax, (SD_xy[0] + w-0.6, SD_xy[1] + h/2), (V_xy[0] + w/2 + 0.2, SD_xy[1] + h/2))
    draw_arrow(ax, (V_xy[0] + w/2 + 0.2, SD_xy[1] + h/2), (V_xy[0] + w/2 + 0.2, V_xy[1]))
    
    # Validate -> SLA
    draw_arrow(ax, (V_xy[0] + w, cy), (SLA_xy[0], cy))
    
    # SLA -> Action (Log)
    draw_line(ax, (SLA_xy[0] + SLA_w/2, SLA_xy[1] + SLA_h), (SLA_xy[0] + SLA_w/2, Log_xy[1] + h/2))
    draw_arrow(ax, (SLA_xy[0] + SLA_w/2, Log_xy[1] + h/2), (Log_xy[0] + w, Log_xy[1] + h/2))
    add_label(ax, (SLA_xy[0] + SLA_w/2, SLA_xy[1] + SLA_h), (SLA_xy[0] + SLA_w/2, Log_xy[1] + h/2), 'Yes')
    
    # Log -> Monitor (Feedback loop - violation)
    draw_line(ax, (Log_xy[0], Log_xy[1] + h/2), (Log_xy[0] - 1.0, Log_xy[1] + h/2))
    draw_line(ax, (Log_xy[0] - 1.0, Log_xy[1] + h/2), (Log_xy[0] - 1.0, 9.3))
    draw_line(ax, (Log_xy[0] - 1.0, 9.3), (M_xy[0] + w/2 + 0.5, 9.3)) # across top
    draw_arrow(ax, (M_xy[0] + w/2 + 0.5, 9.3), (M_xy[0] + w/2 + 0.5, M_xy[1] + h))
    add_label(ax, (14.0, 9.3), (14.0, 9.3), 'Feedback Loop (SLA Violation)')
    
    # SLA -> Feedback (No)
    # Route below everything
    draw_line(ax, (SLA_xy[0] + SLA_w/2, SLA_xy[1]), (SLA_xy[0] + SLA_w/2, 0.2))
    add_label(ax, (SLA_xy[0] + SLA_w/2, SLA_xy[1]), (SLA_xy[0] + SLA_w/2, 0.2), 'No (Stable)')
    draw_line(ax, (SLA_xy[0] + SLA_w/2, 0.2), (M_xy[0] + w/2 - 0.5, 0.2))
    draw_arrow(ax, (M_xy[0] + w/2 - 0.5, 0.2), (M_xy[0] + w/2 - 0.5, M_xy[1]))
    add_label(ax, (14.0, 0.2), (14.0, 0.2), 'Feedback Loop (Stable System)')
    
    plt.tight_layout()
    p = os.path.join(fig_dir, 'fig_system_workflow_horizontal.png')
    plt.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Generated {p}")
    
    p2 = os.path.join(fig_dir, 'fig_system_workflow.png')
    plt.savefig(p2, dpi=300, bbox_inches='tight', facecolor='white')

if __name__ == '__main__':
    main()
