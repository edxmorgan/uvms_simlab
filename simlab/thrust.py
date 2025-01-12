def pwm_to_thrust(pwm):
    """Transform pwm to thruster value
    The equation come from:
        https://colab.research.google.com/notebook#fileId=1CEDW9ONTJ8Aik-HVsqck8Y_EcHYLg0zK

    Args:
        pwm (int): pwm value

    Returns:
        float: Thrust value
    """
    return -3.04338931856672e-13*pwm**5 \
        + 2.27813523978448e-9*pwm**4 \
        - 6.73710647138884e-6*pwm**3 \
        + 0.00983670053385902*pwm**2 \
        - 7.08023833982539*pwm \
        + 2003.55692021905