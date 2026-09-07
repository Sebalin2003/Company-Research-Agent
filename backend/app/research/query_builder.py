from __future__ import annotations

from backend.app.domain.reports import EvidenceTopic
from backend.app.research.types import SearchQuery


def build_company_research_queries(company_name: str) -> list[SearchQuery]:
    return [
        *build_fast_company_research_queries(company_name),
        *build_deepening_company_research_queries(company_name),
    ]


def build_fast_company_research_queries(company_name: str) -> list[SearchQuery]:
    display = company_name.strip()
    return [
        SearchQuery(EvidenceTopic.business, f"{display} sitio oficial que hace"),
        SearchQuery(EvidenceTopic.argentina_presence, f"{display} Argentina oficinas empleo"),
        SearchQuery(EvidenceTopic.employees, f"{display} empleados LinkedIn"),
        SearchQuery(EvidenceTopic.salary, f"{display} sueldo IT trainee Argentina"),
        SearchQuery(EvidenceTopic.salary, f"{display} sueldos Argentina Glassdoor"),
        SearchQuery(EvidenceTopic.salary, f"{display} sueldos OpenQube"),
        SearchQuery(EvidenceTopic.benefits, f"{display} beneficios empleados Argentina"),
        SearchQuery(EvidenceTopic.culture, f"{display} opiniones empleados cultura Glassdoor"),
        SearchQuery(EvidenceTopic.interview_process, f"{display} proceso entrevista preguntas"),
        SearchQuery(EvidenceTopic.interview_questions, f"{display} interview questions"),
        SearchQuery(
            EvidenceTopic.open_roles,
            f"{display} carreras empleos Argentina LinkedIn Computrabajo Bumeran ZonaJobs",
        ),
    ]


def build_deepening_company_research_queries(company_name: str) -> list[SearchQuery]:
    display = company_name.strip()
    return [
        SearchQuery(EvidenceTopic.business, f"{display} about company business"),
        SearchQuery(EvidenceTopic.argentina_presence, f"{display} direccion oficinas Argentina"),
        SearchQuery(EvidenceTopic.argentina_presence, f"{display} sede Buenos Aires direccion"),
        SearchQuery(EvidenceTopic.argentina_presence, f"{display} contacto Argentina domicilio"),
        SearchQuery(EvidenceTopic.employees, f"{display} cantidad de empleados"),
        SearchQuery(EvidenceTopic.employees, f"{display} total employees company profile"),
        SearchQuery(EvidenceTopic.employees, f"{display} employees LinkedIn company"),
        SearchQuery(EvidenceTopic.salary, f"{display} sueldos Argentina"),
        SearchQuery(EvidenceTopic.salary, f"{display} rango salarial Argentina"),
        SearchQuery(EvidenceTopic.salary, f"{display} salario mensual Argentina"),
        SearchQuery(EvidenceTopic.salary, f"{display} sueldo pasante IT Argentina"),
        SearchQuery(EvidenceTopic.salary, f"{display} sueldo internship IT Argentina"),
        SearchQuery(EvidenceTopic.salary, f"{display} sueldo desarrollador junior Argentina"),
        SearchQuery(EvidenceTopic.salary, f"{display} salario junior IT Argentina"),
        SearchQuery(EvidenceTopic.salary, f"{display} trainee program salario Argentina"),
        SearchQuery(EvidenceTopic.salary, f"{display} sueldos trainee junior Glassdoor Argentina"),
        SearchQuery(EvidenceTopic.salary, f"{display} salarios Argentina Indeed"),
        SearchQuery(EvidenceTopic.salary, f"{display} salario junior IT Indeed Argentina"),
        SearchQuery(EvidenceTopic.salary, f"{display} sueldos junior trainee OpenQube"),
        SearchQuery(EvidenceTopic.salary, f"{display} remuneracion empleo Argentina"),
        SearchQuery(EvidenceTopic.benefits, f"{display} obra social prepaga beneficios"),
        SearchQuery(EvidenceTopic.culture, f"{display} reviews empleados Argentina"),
        SearchQuery(EvidenceTopic.interview_process, f"{display} etapas entrevista seleccion"),
        SearchQuery(EvidenceTopic.interview_process, f"{display} dificultad entrevista Glassdoor"),
        SearchQuery(EvidenceTopic.interview_questions, f"{display} preguntas entrevista Glassdoor"),
        SearchQuery(EvidenceTopic.open_roles, f"{display} jobs careers Argentina"),
        SearchQuery(
            EvidenceTopic.open_roles,
            f"{display} vacantes Argentina LinkedIn Computrabajo Bumeran ZonaJobs Indeed Get on Board Portal Empleo",
        ),
    ]


def build_official_site_query(company_name: str) -> SearchQuery:
    from backend.app.domain.companies import normalize_company_name

    return SearchQuery(
        topic=EvidenceTopic.business,
        text=f"{normalize_company_name(company_name)} sitio oficial",
    )
