"""Scenario role-play library.

Each scenario is a real-life situation where the LLM plays a character (barista,
clerk, ...) and the learner has a goal to accomplish. Definitions are language-
agnostic: UI text is English, but the role-play itself happens in the configured
target language (the model adapts). Kept generic/portable so they work for any
language and country.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    id: str
    emoji: str
    title: str  # short label for the picker
    setting: str  # where it happens (context for the model + the UI)
    character: str  # the role the LLM plays
    goal: str  # what the learner must accomplish (drives the completion check)


SCENARIOS: list[Scenario] = [
    Scenario(
        id="cafe",
        emoji="☕",
        title="At the café",
        setting="a busy café counter",
        character="a friendly barista",
        goal="order a coffee and something to eat, then ask for the bill and pay",
    ),
    Scenario(
        id="restaurant",
        emoji="🍽️",
        title="At a restaurant",
        setting="a casual restaurant at dinner time",
        character="a waiter",
        goal="get a table for one, order a starter and a main course, and ask for the bill",
    ),
    Scenario(
        id="bank",
        emoji="🏦",
        title="Opening a bank account",
        setting="a bank branch",
        character="a bank clerk",
        goal="explain you're a new student in town and open a basic account, asking about fees and documents",
    ),
    Scenario(
        id="apartment",
        emoji="🏠",
        title="Apartment viewing",
        setting="a flat you're viewing to rent",
        character="a landlord showing the apartment",
        goal="ask about the rent, the rooms, and what's included, and say whether you're interested",
    ),
    Scenario(
        id="university",
        emoji="🎓",
        title="University registration",
        setting="a university's international students office",
        character="an administrator at the registration desk",
        goal="register for your exchange, ask which documents you need and about deadlines",
    ),
    Scenario(
        id="pharmacy",
        emoji="💊",
        title="At the pharmacy",
        setting="a pharmacy",
        character="a pharmacist",
        goal="describe a mild symptom (a headache and sore throat) and ask for something to help",
    ),
    Scenario(
        id="directions",
        emoji="🧭",
        title="Asking for directions",
        setting="a street corner in an unfamiliar neighborhood",
        character="a helpful passer-by",
        goal="explain you're lost and ask how to get to the train station on foot",
    ),
    Scenario(
        id="meeting",
        emoji="🎉",
        title="Meeting new people",
        setting="a relaxed social gathering with other students",
        character="another student you've just met",
        goal="introduce yourself, ask about them, and find something you have in common",
    ),
    Scenario(
        id="market",
        emoji="🛒",
        title="At the market",
        setting="a fresh food market stall",
        character="a market vendor",
        goal="buy some fruit and vegetables by quantity, ask the price, and pay",
    ),
    Scenario(
        id="ticket",
        emoji="🚆",
        title="Buying a train ticket",
        setting="a train station ticket counter",
        character="a ticket agent",
        goal="buy a ticket to a nearby city, choosing a departure time and asking the price",
    ),
]

_BY_ID = {s.id: s for s in SCENARIOS}


def get(scenario_id: str) -> Scenario | None:
    return _BY_ID.get((scenario_id or "").strip())
