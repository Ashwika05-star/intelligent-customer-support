import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up figure
fig, ax = plt.subplots(figsize=(10, 13), dpi=300)
ax.set_xlim(-6.5, 6.5)
ax.set_ylim(-14.5, 0.8)
ax.axis("off")

# Styles
box_props = dict(boxstyle="round,pad=0.4,rounding_size=0.2", ec="#2c3e50", lw=1.6)
diamond_props = dict(boxstyle="square,pad=0.4", ec="#2c3e50", lw=1.6)

def draw_box(x, y, w, h, text, bg, subtitle=""):
    rect = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.15",
        fc=bg, ec="#2c3e50", lw=1.6
    )
    ax.add_patch(rect)
    if subtitle:
        ax.text(x, y + 0.12, text, ha="center", va="center", fontsize=10, weight="bold", color="#1a252f")
        ax.text(x, y - 0.18, subtitle, ha="center", va="center", fontsize=8.5, color="#34495e")
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=10, weight="bold", color="#1a252f")

def draw_diamond(x, y, w, h, text, bg, subtitle=""):
    diamond = patches.RegularPolygon(
        (x, y), numVertices=4, radius=w/1.4,
        fc=bg, ec="#2c3e50", lw=1.6
    )
    ax.add_patch(diamond)
    if subtitle:
        ax.text(x, y + 0.12, text, ha="center", va="center", fontsize=9.5, weight="bold", color="#1a252f")
        ax.text(x, y - 0.18, subtitle, ha="center", va="center", fontsize=8, color="#34495e")
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=9.5, weight="bold", color="#1a252f")

def draw_arrow(x1, y1, x2, y2, label="", color="#2c3e50", dashed=False):
    ls = "--" if dashed else "-"
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->,head_width=0.28,head_length=0.35", color=color, lw=1.8, ls=ls)
    )
    if label:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(
            mid_x, mid_y, label, ha="center", va="center", fontsize=8, weight="bold",
            color=color, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none")
        )

# 1. Main Pipeline Nodes
draw_box(0, 0, 3.2, 0.7, "Customer Query (q_t)", "#dbeafe")
draw_box(0, -1.8, 3.4, 0.8, "1. Intent Classifier", "#fed7aa", "Category & Sentiment Detection")
draw_box(0, -3.8, 3.4, 0.8, "2. Query Rewriter", "#fed7aa", "Multi-Turn Coreference Resolution")
draw_diamond(0, -5.6, 1.2, 0.8, "Domain?", "#fef08a")

draw_box(-2.3, -7.4, 2.9, 0.8, "3A. Policy Retriever", "#bbf7d0", "Returns, Shipping, Warranty")
draw_box(2.3, -7.4, 2.9, 0.8, "3B. Manual Retriever", "#bbf7d0", "Hardware Specs, Codes")

draw_diamond(0, -9.6, 1.4, 0.9, "4. CRAG Grader", "#fecaca", "Score >= 0.30?")
draw_box(0, -11.6, 3.4, 0.8, "5. Grounded Generator", "#e9d5ff", "Answer with Citations")
draw_box(0, -13.2, 3.2, 0.7, "Deliver Response", "#dbeafe")

# 2. Side Branches
# Left: Human Escalation
draw_box(-4.9, -5.6, 2.8, 0.8, "Human Escalation", "#fca5a5", "Priority Ticket (HITL)")

# Right: Chitchat and Order DB Tool
draw_box(4.9, -1.8, 2.7, 0.8, "Chitchat Node", "#e2e8f0", "Greetings / Pleasantries")
draw_box(4.9, -4.5, 2.7, 0.8, "Order DB Tool", "#99f6e4", "Status & 30-Day Return Check")

# 3. Main Center Arrows
draw_arrow(0, -0.35, 0, -1.4)
draw_arrow(0, -2.2, 0, -3.4, label="support")
draw_arrow(0, -4.2, 0, -5.0)

# Domain to retrievers
ax.plot([0, -2.3, -2.3], [-6.2, -6.2, -7.0], color="#2c3e50", lw=1.8)
ax.annotate("", xy=(-2.3, -7.0), xytext=(-2.3, -6.9), arrowprops=dict(arrowstyle="->,head_width=0.28,head_length=0.35", color="#2c3e50", lw=1.8))
ax.text(-1.2, -6.0, "policy", ha="center", va="bottom", fontsize=8, weight="bold", color="#2c3e50", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

ax.plot([0, 2.3, 2.3], [-6.2, -6.2, -7.0], color="#2c3e50", lw=1.8)
ax.annotate("", xy=(2.3, -7.0), xytext=(2.3, -6.9), arrowprops=dict(arrowstyle="->,head_width=0.28,head_length=0.35", color="#2c3e50", lw=1.8))
ax.text(1.2, -6.0, "manual", ha="center", va="bottom", fontsize=8, weight="bold", color="#2c3e50", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

# Retrievers to Grader
ax.plot([-2.3, -2.3, -0.8], [-7.8, -8.6, -9.6], color="#2c3e50", lw=1.8)
ax.annotate("", xy=(-0.7, -9.6), xytext=(-0.8, -9.6), arrowprops=dict(arrowstyle="->,head_width=0.28,head_length=0.35", color="#2c3e50", lw=1.8))

ax.plot([2.3, 2.3, 0.8], [-7.8, -8.6, -9.6], color="#2c3e50", lw=1.8)
ax.annotate("", xy=(0.7, -9.6), xytext=(0.8, -9.6), arrowprops=dict(arrowstyle="->,head_width=0.28,head_length=0.35", color="#2c3e50", lw=1.8))

# Grader to Generator & Deliver
draw_arrow(0, -10.3, 0, -11.2, label="Pass (>= 0.30)")
draw_arrow(0, -12.0, 0, -12.85)

# 4. Left Escalation Routing (Zero Overlap)
# Classifier to Escalator
ax.plot([-1.7, -4.9, -4.9], [-1.8, -1.8, -5.2], color="#2c3e50", lw=1.8)
ax.annotate("", xy=(-4.9, -5.2), xytext=(-4.9, -5.1), arrowprops=dict(arrowstyle="->,head_width=0.28,head_length=0.35", color="#2c3e50", lw=1.8))
ax.text(-3.3, -1.65, "anger / human", ha="center", va="bottom", fontsize=8, weight="bold", color="#2c3e50", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

# Grader Fail to Escalator
ax.plot([-0.7, -4.9, -4.9], [-9.6, -9.6, -6.0], color="#2c3e50", lw=1.8)
ax.annotate("", xy=(-4.9, -6.0), xytext=(-4.9, -6.1), arrowprops=dict(arrowstyle="->,head_width=0.28,head_length=0.35", color="#2c3e50", lw=1.8))
ax.text(-3.3, -9.45, "Fail (retry >= 1)", ha="center", va="bottom", fontsize=8, weight="bold", color="#2c3e50", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

# Escalator to Deliver
ax.plot([-4.9, -5.8, -5.8, -1.6], [-6.0, -6.0, -13.2, -13.2], color="#2c3e50", lw=1.8)
ax.annotate("", xy=(-1.6, -13.2), xytext=(-1.7, -13.2), arrowprops=dict(arrowstyle="->,head_width=0.28,head_length=0.35", color="#2c3e50", lw=1.8))

# 5. Right Side Routing (Zero Overlap)
# Classifier to Chitchat
draw_arrow(1.7, -1.8, 3.55, -1.8, label="chitchat")
ax.plot([4.9, 6.0, 6.0, 1.6], [-2.2, -2.2, -13.2, -13.2], color="#2c3e50", lw=1.8)
ax.annotate("", xy=(1.6, -13.2), xytext=(1.7, -13.2), arrowprops=dict(arrowstyle="->,head_width=0.28,head_length=0.35", color="#2c3e50", lw=1.8))

# Classifier to Order DB Tool
ax.plot([1.7, 2.8, 2.8, 3.55], [-2.0, -2.0, -4.5, -4.5], color="#2c3e50", lw=1.8)
ax.annotate("", xy=(3.55, -4.5), xytext=(3.45, -4.5), arrowprops=dict(arrowstyle="->,head_width=0.28,head_length=0.35", color="#2c3e50", lw=1.8))
ax.text(2.8, -3.2, "order/return", ha="center", va="center", fontsize=8, weight="bold", color="#2c3e50", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

# Order DB Tool to Grader
ax.plot([4.9, 4.9, 0.8], [-4.9, -8.6, -9.6], color="#2c3e50", lw=1.8)
ax.annotate("", xy=(0.7, -9.6), xytext=(0.8, -9.6), arrowprops=dict(arrowstyle="->,head_width=0.28,head_length=0.35", color="#2c3e50", lw=1.8))

# 6. Red Retry Loop (Far Right Isolated Lane - ZERO OVERLAP)
ax.plot([0.5, 0.5, 6.4, 6.4, 1.7], [-10.1, -10.6, -10.6, -3.4, -3.4], color="#dc2626", lw=1.8, ls="--")
ax.annotate("", xy=(1.7, -3.4), xytext=(1.8, -3.4), arrowprops=dict(arrowstyle="->,head_width=0.28,head_length=0.35", color="#dc2626", lw=1.8, ls="--"))
ax.text(6.4, -7.0, "Retry (< 1)", ha="center", va="center", fontsize=8.5, weight="bold", color="#dc2626", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#dc2626"))

plt.tight_layout()
plt.savefig("/Users/apar/.gemini/antigravity/scratch/intelligent-customer-support/flowchart.png", dpi=300, bbox_inches="tight")
plt.savefig("/Users/apar/.gemini/antigravity/brain/a28296a6-c009-41a0-9d5d-e0b2a03df43c/flowchart.png", dpi=300, bbox_inches="tight")
print("Flowchart generated successfully!")
