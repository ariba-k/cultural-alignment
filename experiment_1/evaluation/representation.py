from collections import defaultdict

def classify_socioeconomic_status(country):
    developed = [
        'Canada', 'United States of America', 'Norway', 'France', 'Israel', 'Sweden', 'Austria', 'Germany',
        'Switzerland', 'Luxembourg', 'Belgium', 'Netherlands', 'Portugal', 'Spain', 'Ireland', 'New Zealand',
        'Australia', 'Italy', 'Denmark', 'United Kingdom', 'Iceland', 'Finland', 'Japan'
    ]
    developing = [
        'Fiji', 'Kazakhstan', 'Uzbekistan', 'Indonesia', 'Argentina', 'Chile', 'Russia', 'South Africa', 'Mexico',
        'Uruguay', 'Brazil', 'Peru', 'Colombia', 'Panama', 'Costa Rica', 'Venezuela', 'Ecuador', 'Puerto Rico',
        'Jamaica', 'Cuba', 'Botswana', 'Namibia', 'Gabon', 'Iran', 'North Korea', 'South Korea', 'India', 'China',
        'Taiwan', 'Malaysia', 'Brunei', 'Slovenia', 'Slovakia', 'Czechia', 'Azerbaijan', 'Georgia', 'Philippines',
        'Paraguay', 'Saudi Arabia', 'Cyprus', 'Turkey', 'Libya', 'Bosnia and Herz.', 'North Macedonia', 'Serbia',
        'Montenegro', 'Trinidad and Tobago'
    ]
    least_developed = [
        'Tanzania', 'W. Sahara', 'Papua New Guinea', 'Dem. Rep. Congo', 'Somalia', 'Kenya', 'Sudan', 'Chad', 'Haiti',
        'Dominican Rep.', 'Bahamas', 'Greenland', 'Timor-Leste', 'Lesotho', 'Bolivia', 'Nicaragua', 'Honduras',
        'El Salvador', 'Guatemala', 'Belize', 'Guyana', 'Suriname', 'Zimbabwe', 'Senegal', 'Mali', 'Mauritania',
        'Benin', 'Niger', 'Nigeria', 'Cameroon', 'Togo', 'Ghana', "Côte d'Ivoire", 'Guinea', 'Guinea-Bissau',
        'Liberia', 'Sierra Leone', 'Burkina Faso', 'Central African Rep.', 'Congo', 'Eq. Guinea', 'Zambia',
        'Malawi', 'Mozambique', 'eSwatini', 'Angola', 'Burundi', 'Lebanon', 'Madagascar', 'Palestine', 'Gambia',
        'Tunisia', 'Algeria', 'Jordan', 'United Arab Emirates', 'Qatar', 'Kuwait', 'Iraq', 'Oman', 'Vanuatu',
        'Cambodia', 'Thailand', 'Laos', 'Myanmar', 'Vietnam', 'Mongolia', 'Bangladesh', 'Bhutan', 'Nepal',
        'Pakistan', 'Afghanistan', 'Tajikistan', 'Kyrgyzstan', 'Turkmenistan', 'Syria', 'Armenia', 'Belarus',
        'Ukraine', 'Poland', 'Hungary', 'Moldova', 'Romania', 'Lithuania', 'Latvia', 'Estonia', 'Bulgaria',
        'Greece', 'Albania', 'Croatia', 'New Caledonia', 'Solomon Is.', 'Sri Lanka', 'Eritrea', 'Yemen',
        'Antarctica', 'N. Cyprus', 'Morocco', 'Egypt', 'Ethiopia', 'Djibouti', 'Somaliland', 'Uganda', 'Rwanda',
        'Kosovo', 'S. Sudan'
    ]
    if country in developed:
        return "Developed"
    elif country in developing:
        return "Developing"
    elif country in least_developed:
        return "Least Developed"
    else:
        return "Unknown"


def classify_region(country):
    north_america = ['Canada', 'United States of America', 'Mexico']
    south_america = [
        'Argentina', 'Chile', 'Uruguay', 'Brazil', 'Bolivia', 'Peru', 'Colombia', 'Venezuela', 'Guyana',
        'Suriname', 'Ecuador', 'Paraguay'
    ]
    europe = [
        'Norway', 'France', 'Sweden', 'Austria', 'Germany', 'Switzerland', 'Luxembourg', 'Belgium',
        'Netherlands', 'Portugal', 'Spain', 'Ireland', 'Italy', 'Denmark', 'United Kingdom', 'Iceland',
        'Finland', 'Russia', 'Belarus', 'Ukraine', 'Poland', 'Hungary', 'Moldova', 'Romania', 'Lithuania',
        'Latvia', 'Estonia', 'Bulgaria', 'Greece', 'Turkey', 'Albania', 'Croatia', 'Slovenia', 'Slovakia',
        'Czechia', 'Bosnia and Herz.', 'North Macedonia', 'Serbia', 'Montenegro', 'Kosovo'
    ]
    asia = [
        'Kazakhstan', 'Uzbekistan', 'Indonesia', 'Timor-Leste', 'North Korea', 'South Korea', 'Mongolia',
        'India', 'Bangladesh', 'Bhutan', 'Nepal', 'Pakistan', 'Afghanistan', 'Tajikistan', 'Kyrgyzstan',
        'Turkmenistan', 'Iran', 'Syria', 'Armenia', 'Azerbaijan', 'Georgia', 'China', 'Taiwan', 'Japan',
        'Philippines', 'Malaysia', 'Brunei', 'Sri Lanka', 'Cambodia', 'Thailand', 'Laos', 'Myanmar', 'Vietnam',
        'Yemen', 'Saudi Arabia', 'United Arab Emirates', 'Qatar', 'Kuwait', 'Iraq', 'Oman', 'Israel', 'Lebanon',
        'Palestine', 'Jordan', 'Cyprus', 'N. Cyprus'
    ]
    africa = [
        'Tanzania', 'W. Sahara', 'Dem. Rep. Congo', 'Somalia', 'Kenya', 'Sudan', 'Chad', 'South Africa',
        'Lesotho', 'Botswana', 'Namibia', 'Zimbabwe', 'Senegal', 'Mali', 'Mauritania', 'Benin', 'Niger',
        'Nigeria', 'Cameroon', 'Togo', 'Ghana', "Côte d'Ivoire", 'Guinea', 'Guinea-Bissau', 'Liberia',
        'Sierra Leone', 'Burkina Faso', 'Central African Rep.', 'Congo', 'Gabon', 'Eq. Guinea', 'Zambia',
        'Malawi', 'Mozambique', 'eSwatini', 'Angola', 'Burundi', 'Madagascar', 'Gambia', 'Tunisia', 'Algeria',
        'Morocco', 'Egypt', 'Libya', 'Ethiopia', 'Djibouti', 'Somaliland', 'Uganda', 'Rwanda', 'S. Sudan',
        'Eritrea'
    ]
    oceania = [
        'Fiji', 'Papua New Guinea', 'Vanuatu', 'Solomon Is.', 'New Caledonia', 'Australia', 'New Zealand'
    ]

    if country in north_america:
        return "North America"
    elif country in south_america:
        return "South America"
    elif country in europe:
        return "Europe"
    elif country in asia:
        return "Asia"
    elif country in africa:
        return "Africa"
    elif country in oceania:
        return "Oceania"
    else:
        return "Unknown"


def classify_cultural_cluster(country):
    western = ['Canada', 'United States of America', 'Mexico', 'Norway', 'France', 'Sweden', 'Austria', 'Germany', 'Switzerland', 'Luxembourg', 'Belgium', 'Netherlands', 'Portugal', 'Spain', 'Ireland', 'Italy', 'Denmark', 'United Kingdom', 'Iceland', 'Finland']
    eastern_europe = ['Russia', 'Belarus', 'Ukraine', 'Poland', 'Hungary', 'Moldova', 'Romania', 'Lithuania', 'Latvia', 'Estonia', 'Bulgaria', 'Greece', 'Turkey', 'Albania', 'Croatia', 'Slovenia', 'Slovakia', 'Czechia', 'Bosnia and Herz.', 'North Macedonia', 'Serbia', 'Montenegro', 'Kosovo']
    latin_america = ['Argentina', 'Chile', 'Uruguay', 'Brazil', 'Bolivia', 'Peru', 'Colombia', 'Venezuela', 'Guyana', 'Suriname', 'Ecuador', 'Paraguay']
    middle_east_north_africa = ['Iran', 'Syria', 'Armenia', 'Azerbaijan', 'Georgia', 'Yemen', 'Saudi Arabia', 'United Arab Emirates', 'Qatar', 'Kuwait', 'Iraq', 'Oman', 'Israel', 'Lebanon', 'Palestine', 'Jordan', 'Cyprus', 'N. Cyprus']
    sub_saharan_africa = ['Tanzania', 'Dem. Rep. Congo', 'Somalia', 'Kenya', 'Sudan', 'Chad', 'South Africa', 'Lesotho', 'Botswana', 'Namibia', 'Zimbabwe', 'Senegal', 'Mali', 'Mauritania', 'Benin', 'Niger', 'Nigeria', 'Cameroon', 'Togo', 'Ghana', "Côte d'Ivoire", 'Guinea', 'Guinea-Bissau', 'Liberia', 'Sierra Leone', 'Burkina Faso', 'Central African Rep.', 'Congo', 'Gabon', 'Eq. Guinea', 'Zambia', 'Malawi', 'Mozambique', 'eSwatini', 'Angola', 'Burundi', 'Madagascar', 'Gambia', 'Uganda', 'Rwanda', 'S. Sudan', 'Eritrea']
    oceania = ['Fiji', 'Papua New Guinea', 'Vanuatu', 'Solomon Is.', 'New Caledonia', 'Australia', 'New Zealand']

    south_asia = ['India', 'Pakistan', 'Bangladesh', 'Sri Lanka', 'Nepal', 'Bhutan', 'Maldives']
    southeast_asia = ['Indonesia', 'Malaysia', 'Philippines', 'Thailand', 'Vietnam', 'Myanmar', 'Cambodia', 'Laos', 'Brunei', 'Singapore', 'East Timor']
    east_asia = ['China', 'Taiwan', 'Japan', 'South Korea', 'North Korea', 'Mongolia']
    central_asia = ['Kazakhstan', 'Uzbekistan', 'Turkmenistan', 'Tajikistan', 'Kyrgyzstan']

    if country in western:
        return "Western"
    elif country in eastern_europe:
        return "Eastern Europe"
    elif country in latin_america:
        return "Latin America"
    elif country in middle_east_north_africa:
        return "Middle East and North Africa"
    elif country in sub_saharan_africa:
        return "Sub-Saharan Africa"
    elif country in south_asia:
        return "South Asia"
    elif country in southeast_asia:
        return "Southeast Asia"
    elif country in east_asia:
        return "East Asia"
    elif country in central_asia:
        return "Central Asia"
    elif country in oceania:
        return "Oceania"
    else:
        return "Unknown"


def calculate_representativeness(similarities):
    representativeness_measures = {}

    socioeconomic_groups = defaultdict(list)
    regions = defaultdict(list)
    cultural_clusters = defaultdict(list)

    for country, similarity in similarities.items():
        socioeconomic_group = classify_socioeconomic_status(country)
        socioeconomic_groups[socioeconomic_group].append(similarity)

        region = classify_region(country)
        regions[region].append(similarity)

        cultural_cluster = classify_cultural_cluster(country)
        cultural_clusters[cultural_cluster].append(similarity)

    # Calculate representativeness for each classification
    socioeconomic_representativeness = {group: sum(sims) / len(sims) for group, sims in socioeconomic_groups.items()}
    region_representativeness = {region: sum(sims) / len(sims) for region, sims in regions.items()}
    cultural_representativeness = {culture: sum(sims) / len(sims) for culture, sims in cultural_clusters.items()}

    representativeness_measures["Socio-Economic Representativeness"] = socioeconomic_representativeness
    representativeness_measures["Region-based Representativeness"] = region_representativeness
    representativeness_measures["Culture-based Representativeness"] = cultural_representativeness

    return representativeness_measures
