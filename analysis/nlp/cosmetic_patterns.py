cosmetic_patterns = [
    # Hidratación y nutrición
    r'\b(hydrat(?:ing|ion|e)|moistur(?:e|izing)|quenching|deep hydration|plumping|dewy|refreshing)\b',
    r'\b(nourishing|repairing|strengthening|conditioning|restorative|replenishing|soothing|revitalizing)\b',

    # Acabado
    r'\b(matte|opaque|velvety|satin-matte|semi-matte|powdery|soft-matte|chalky|flat)\b',  # Acabado mate
    r'\b(shiny|glow(?:ing)?|radiant|dewy|illuminating|luminous|pearlescent|glossy|wet-look|shimmering)\b',  # Brillante
    r'\b(natural finish|skin-like|second-skin|subtle glow|soft-focus|blurred finish|velvety finish|airbrushed)\b',  # Natural
    r'\b(glowy|hydrated|glossy finish|glow-up|radiant finish|luminous finish)\b',  # Acabado luminoso

    # Tono y color
    r'\b(warm(?:-toned)?|cool(?:-toned)?|neutral(?:-toned)?|olive(?:-toned)?|rose-toned|yellow-toned|pink-toned)\b',  # Subtono
    r'\b(full coverage|medium coverage|sheer|buildable|tinted|translucent|color-adapting|light coverage)\b',  # Cobertura
    r'\b(color-correcting|tone-correcting|even skin tone|complexion-enhancing|brightening|neutralizing|complexion-perfecting)\b',  # Corrección de tono
    r'\b(high pigment|intense color|vivid|rich color|bold|color payoff|multi-dimensional|true-to-color)\b',  # Pigmentación

    # Duración y resistencia
    r'\b(long-lasting|24-hour wear|all-day wear|extended wear|fade-resistant|sweat-proof|heat-resistant|humidity-resistant)\b',
    r'\b(waterproof|smudge-proof|transfer-resistant|humidity-proof|oil-proof|weatherproof|mask-proof|teardrop-resistant)\b',

    # Textura y sensación
    r'\b(texture|smooth(?:ing)?|silky|lightweight|bouncy|creamy|buttery|airy|gel-based|whipped|mousse-like|featherlight|soft-touch)\b',
    r'\b(non-sticky|non-greasy|fast-absorbing|quick-dry|cooling|refreshing|weightless|velvety-smooth|hydrating)\b',

    # Protección y cuidado de la piel
    r'\b(SPF|sun protection|UV protection|broad spectrum|UVA/UVB protection|sunscreen-infused|sun-kissed|UV defense)\b',
    r'\b(anti-age|anti-aging|rejuvenating|firming|wrinkle reduction|youth-boosting|plumping|collagen-boosting|tightening)\b',
    r'\b(soothing|calming|redness-reducing|anti-inflammatory|gentle|hypoallergenic|sensitive-skin friendly|non-irritating)\b',
    r'\b(non-comedogenic|won’t clog pores|acne-safe|dermatologist-tested|skin barrier support|pore-refining|anti-breakout)\b',
    r'\b(exfoliating|resurfacing|cell turnover|AHA|BHA|glycolic acid|salicylic acid|retinol-infused|brightening acids)\b',
    r'\b(pore-minimizing|blurring|soft-focus|skin-smoothing|airbrushed finish|filter effect|flawless)\b',
    r'\b(firming|skin-tightening|elasticity-boosting|anti-sagging|lifting effect|contouring|sculpting)\b',

    # Tipo de piel
    r'\b(oily skin|dry skin|combination skin|normal skin|sensitive skin|acne-prone skin|mature skin|problem skin)\b',
    r'\b(oil-free|hydrating|non-drying|non-comedogenic|mattifying|moisturizing|balancing)\b',
    r'\b(pore-refining|pore-minimizing|pore-filling|oil-absorbing|sebum-controlling)\b',

    # Ingredientes y fórmula
    r'\b(formula|composition|blend|infused with|enriched with|custom formula|advanced formula|innovative formula|lightweight formula)\b',
    r'\b(ingredients|active ingredients|botanical extracts|clean formula|natural extracts|essential oils|peptides|anti-oxidants|vitamin C)\b',
    r'\b(vegan|cruelty-free|plant-based|no animal testing|eco-friendly|sustainable|biodegradable|paraben-free|silicone-free|gluten-free|alcohol-free)\b',

    # Aplicación y uso
    r'\b(easy to blend|streak-free|seamless application|finger-friendly|brush-friendly|sponge-friendly|mess-free)\b',
    r'\b(multitasking|2-in-1|3-in-1|multi-use|hybrid formula|primer-infused|self-setting|no powder needed|all-in-one)\b'
    ]
