"""Datasheet numbers for edge targets, used only to plug into formulas.

Nothing here is a measurement. Measured numbers stay in each lab's
``artifacts/raw``. Bandwidths are peak bus arithmetic (data rate x width), not
sustained. NPU TOPS are deliberately absent: vendor figures mix dense and
sparse rates and change between SKUs, so a formula that needs one takes it as
an argument.
"""

from __future__ import annotations

from dataclasses import dataclass


class UnknownDevice(KeyError):
    pass


@dataclass(frozen=True)
class Device:
    slug: str
    name: str
    cpu: str
    clock_hz: float
    sram_bytes: int | None
    flash_bytes: int | None
    dram_bw: float | None
    note: str = ""


_DEVICES = (
    Device("nrf52840", "Nordic nRF52840", "Cortex-M4F", 64e6, 256 * 1024, 1024 * 1024, None,
           "Arduino Nano 33 BLE Sense, the HarvardX TinyML kit. No external DRAM."),
    Device("stm32h747", "STMicro STM32H747", "Cortex-M7 + Cortex-M4", 480e6, 1024 * 1024, 2 * 1024 * 1024, None,
           "Arduino Portenta H7 / Nicla Vision class, used by CS249r 2023. SRAM split across domains."),
    Device("pi5", "Raspberry Pi 5", "4x Cortex-A76", 2.4e9, None, None, 4267e6 * 4,
           "LPDDR4X-4267 on a 32-bit bus: 17.1 GB/s peak."),
    Device("orin-nano-8g", "Jetson Orin Nano 8GB", "6x Cortex-A78AE + Ampere GPU", 1.5e9, None, None, 68e9,
           "128-bit LPDDR5, 68 GB/s. The 4GB module has a 64-bit bus."),
)

_BY_SLUG = {d.slug: d for d in _DEVICES}


def get(slug: str) -> Device:
    try:
        return _BY_SLUG[slug]
    except KeyError:
        raise UnknownDevice(slug) from None


def all_devices() -> tuple[Device, ...]:
    return _DEVICES
