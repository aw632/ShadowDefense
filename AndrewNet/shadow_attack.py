from pso import PSO
from torchvision import transforms
from shadow_attack import pre_process_image


def attack(
    attack_image,
    label,
    coords,
    our_model,
    targeted_attack=False,
    physical_attack=False,
    testing=False,
    **parameters,
):
    r"""
    Physical-world adversarial attack by shadow.

    Args:
        attack_image: The image to be attacked.
        label: The ground-truth label of attack_image.
        coords: The coordinates of the points where mask == 1.
        targeted_attack: Targeted / Non-targeted attack.
        physical_attack: Physical / digital attack.

    Returns:
        adv_img: The generated adversarial image.
        succeed: Whether the attack is successful.
        num_query: Number of queries.
    """
    num_query = 0
    succeed = False
    global_best_solution = float("inf")
    global_best_position = None

    # transform = transforms.Compose([transforms.ToTensor()])
    # attack_image = transform(attack_image)

    for attempt in range(5):

        if succeed:
            break

        # if not testing:
        # print(f"try {attempt + 1}:", end=" ")

        polygon = 3
        particle_size = 10
        iter_num = 100
        x_min, x_max = -16, 48
        max_speed = 1.5
        shadow_level = 0.43
        pre_process = transforms.Compose([pre_process_image, transforms.ToTensor()])
        pso = PSO(
            polygon * 2,
            particle_size,
            iter_num,
            x_min,
            x_max,
            max_speed,
            shadow_level,
            attack_image,
            coords,
            our_model,
            targeted_attack,
            physical_attack,
            label,
            pre_process,
            **parameters,
        )
        best_solution, best_pos, succeed, query = (
            pso.update_digital() if not physical_attack else pso.update_physical()
        )

        if targeted_attack:
            best_solution = 1 - best_solution
        # if not testing:
        #     print(
        #         f"Best solution: {best_solution} {'succeed' if succeed else 'failed'}"
        #     )
        if best_solution < global_best_solution:
            global_best_solution = best_solution
            global_best_position = best_pos
        num_query += query

    # adv_image, shadow_area = draw_shadow(
    #     global_best_position, attack_image, coords, shadow_level
    # )
    # adv_image = shadow_edge_blur(adv_image, shadow_area, 3)

    return succeed, num_query
