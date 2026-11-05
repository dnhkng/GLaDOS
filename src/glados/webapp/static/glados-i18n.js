// Console language. The markup and scripts stay in English (as upstream ships them); this layer
// swaps exact English strings for their translation as they are rendered, and back again.
(() => {
  const FR = {
    // Navigation & header
    'Theme': 'Thème', 'Interface theme': "Thème de l'interface", 'Language': 'Langue',
    'Presence screen ↗': 'Écran de présence ↗', 'Full view': 'Vue complète', 'Display': 'Affichage', 'Simple view': 'Vue simple', 'CONNECTING': 'CONNEXION', 'Views': 'Vues',
    'Enrichment Center': 'Accueil', 'Cores': 'Noyaux', 'Test Chamber': 'Mémoire',
    'Neural Buffer': 'Contexte', 'Facility Tools': 'Outils', 'Test Assignments': 'Tâches',
    'Facility Settings': 'Réglages', 'Observation Log': 'Journal', 'Diagnostics': 'Diagnostics',
    'LIVE TELEMETRY LINKED': 'TÉLÉMÉTRIE EN DIRECT', 'LIVE TELEMETRY': 'TÉLÉMÉTRIE EN DIRECT',
    'DISCONNECTED · RETRYING': 'DÉCONNECTÉ · NOUVELLE TENTATIVE', 'DEMO DATA': 'DONNÉES DE DÉMO',
    // Home
    'Aperture Science Enrichment Center. Inspect each core’s activity, controls and reports.':
      "Centre d'enrichissement Aperture Science. Consultez l'activité, les commandes et les rapports de chaque noyau.",
    'Use this device’s microphone': 'Utiliser le micro de cet appareil',
    'Facility observation and control.': 'Observation et contrôle du centre.',
    'GLaDOS eye and optic feed': 'Œil de GLaDOS et flux optique', 'GLaDOS aperture, connecting': 'Œil de GLaDOS, connexion',
    'Connecting': 'Connexion', 'Connecting…': 'Connexion…', 'You': 'Vous', 'No input received yet.': 'Aucune entrée reçue pour le moment.',
    'Replies appear here, including when voice output is muted.': 'Les réponses apparaissent ici, même quand la voix est coupée.',
    'Live optic feed with theme treatment': 'Flux optique en direct avec effet de thème',
    'TEST SUBJECT': 'SUJET DE TEST', 'SUBJECT': 'SUJET', 'MOVEMENT': 'MOUVEMENT', 'MOTION': 'MOUVEMENT',
    'No observation yet.': 'Aucune observation pour le moment.', 'On her mind': 'Ce qui l’occupe',
    'Talk to GLaDOS': 'Parler à GLaDOS', 'Send a message or ask her to work on a task': 'Envoyez un message ou confiez-lui une tâche',
    'What would you like to do?': 'Que voulez-vous faire ?', 'Send message': 'Envoyer',
    'Latest turn': 'Dernier échange', 'Waiting for input': "En attente d'une entrée",
    'Recorded server events: input accepted → playback requested. Excludes endpoint silence and browser audio buffering.':
      'Événements serveur enregistrés : entrée acceptée → lecture demandée. Hors silence de fin de phrase et tampon audio du navigateur.',
    'Timing begins with the next user input.': "Le chronométrage commence à la prochaine entrée.",
    'Telemetry disconnected': 'Télémétrie déconnectée', 'Turn in progress': 'Échange en cours',
    'Routing completed': 'Routage terminé', 'Reply inference admitted': 'Inférence de réponse admise',
    'First response token': 'Premier jeton de réponse', 'First clause to TTS': 'Première phrase vers la synthèse',
    'First audio ready': 'Premier audio prêt', 'Playback requested': 'Lecture demandée',
    'Inference slots': "Emplacements d'inférence", 'Inspect': 'Inspecter', 'Core activity': 'Activité des noyaux',
    'Core': 'Noyau', 'State': 'État', 'Latest update': 'Dernière mise à jour', 'Settings': 'Réglages',
    'Assignments tracked →': 'Missions suivies →', 'Online cores →': 'Noyaux en ligne →',
    'User-facing inferences →': 'Inférences utilisateur →', 'Dual Lane Orchestration': 'Orchestration à deux voies',
    'Priority Lane': 'Voie prioritaire', 'routing and conversation requests': 'routage et conversation',
    'Background Lane': "Voie d'arrière-plan", 'emotion and background cores': 'émotion et noyaux de fond',
    'Engine state': 'État du moteur', 'Microphone': 'Microphone', 'Voice output': 'Voix', 'Camera': 'Caméra',
    'Quiet': 'Silence', 'Pleasure': 'Plaisir', 'Arousal': 'Excitation', 'Dominance': 'Dominance',
    'Edit response instructions': 'Modifier les consignes de réponse', 'Inspect context': 'Inspecter le contexte',
    'Inspect emotion': "Inspecter l'émotion", 'Last user': 'Dernier message', 'Last spoke': 'Dernière prise de parole',
    'Since last user': 'Depuis votre dernier message', 'Since last spoke': 'Depuis sa dernière parole',
    'Test assignments': 'Tâches', 'Create assignment': 'Créer une mission', 'Edit assignment': 'Modifier la mission',
    'Assignments are shared with the Central Core. Ask GLaDOS to work on an open assignment.':
      'Les missions sont partagées avec le noyau central. Demandez à GLaDOS de travailler sur une mission ouverte.',
    'Observation log · live stream': "Journal d'observation · direct", 'open inspector →': "ouvrir l'inspecteur →",
    'Ready when you are.': 'Prête quand vous voulez.', 'Waiting for GLaDOS…': 'En attente de GLaDOS…',
    'Enter a message.': 'Saisissez un message.', 'Message sent to GLaDOS.': 'Message envoyé à GLaDOS.',
    // Cores
    'Central Core': 'Noyau central', 'Central Core · GLaDOS': 'Noyau central · GLaDOS', 'Autonomy Core': "Noyau d'autonomie",
    'Vision Core': 'Noyau Vision', 'Emotion Core': 'Noyau Émotion', 'Memory Core': 'Noyau Mémoire', 'Health Core': 'Noyau Santé',
    'Search Core': 'Noyau Recherche', 'Routing Core': 'Noyau Routage', 'Nexus Core': 'Noyau Nexus', 'Weather Core': 'Noyau Météo',
    'Planning Core': 'Noyau Planification', 'Maintenance Core': 'Noyau Maintenance',
    'Dedicated functions. Shared processing.': 'Fonctions dédiées. Calcul partagé.', 'system diagnostics →': 'diagnostics système →',
    'The Central Core handles your requests. Other cores maintain their own functions, state and reports. Suspend a core to stop future scheduled runs; a current run may finish.':
      "Le noyau central traite vos demandes. Les autres noyaux gèrent leurs propres fonctions, état et rapports. Suspendre un noyau arrête ses prochaines exécutions ; celle en cours peut se terminer.",
    'Suspended': 'Suspendu', 'Waiting for resume': 'En attente de reprise', 'Working': 'En cours', 'Queued': 'En file',
    'Waiting for user response': 'En attente de votre réponse', 'Waiting for a mind worker': "En attente d'un exécuteur",
    'Processing update': 'Traitement de la mise à jour', 'Offline': 'Hors ligne', 'Function disabled': 'Fonction désactivée',
    'Standby': 'En veille', 'On demand': 'À la demande', 'Ready for requests': 'Prêt pour les demandes',
    'Awaiting first report.': 'En attente du premier rapport.', 'Important · review': 'Important · à examiner',
    'Regular update': 'Mise à jour courante', 'Cancel task': 'Annuler la tâche', 'Cancellation requested': 'Annulation demandée',
    'Resume core': 'Reprendre le noyau', 'Suspend core': 'Suspendre le noyau', 'No output yet.': 'Aucune sortie pour le moment.',
    'Schedule updated.': 'Planification mise à jour.', 'One run queued. The result will refresh when it finishes.':
      'Une exécution en file. Le résultat sera actualisé une fois terminée.',
    'On requested research': 'Sur demande de recherche', 'Completed/attempted runs': 'Exécutions réussies/tentées',
    'Research topic': 'Sujet de recherche', 'Waiting for a request': "En attente d'une demande", 'Search rounds': 'Tours de recherche',
    'Cited passages': 'Passages cités', 'Time limit': 'Limite de temps', 'Recent messages kept': 'Messages récents conservés',
    'Raw messages': 'Messages bruts', 'Estimated context': 'Contexte estimé', 'Recall topic': 'Sujet de rappel',
    'Waiting for a topic': "En attente d'un sujet", 'Autonomy Core unavailable.': "Noyau d'autonomie indisponible.",
    'No pending handoff to Central Core.': 'Aucun transfert en attente vers le noyau central.',
    'PAD schedule paused': 'Planification PAD en pause', 'PAD agent active': 'Agent PAD actif',
    'Emotion tracking is disabled in this profile. Enable autonomy.emotion.enabled to use the PAD gauges.':
      "Le suivi des émotions est désactivé dans ce profil. Activez autonomy.emotion.enabled pour utiliser les jauges PAD.",
    // Processing / tools
    'Shared processing · active requests and queues': 'Calcul partagé · demandes actives et files',
    'Cores share the available processing channels. Routing and your replies have priority; background cores use spare capacity.':
      'Les noyaux partagent les canaux de calcul. Le routage et vos réponses sont prioritaires ; les noyaux de fond utilisent la capacité restante.',
    'Facility command queue': 'File de commandes du centre', 'Run a task': 'Lancer une tâche',
    'Fixed, read-only host operations run on the CPU alongside the cores.': "Opérations fixes en lecture seule, exécutées sur le CPU à côté des noyaux.",
    'Waiting for capacity': 'En attente de capacité', 'None': 'Aucun', 'Latest routing decision': 'Dernière décision de routage',
    'No decision yet.': 'Aucune décision pour le moment.', 'No routing decision yet.': 'Aucune décision de routage pour le moment.',
    'capabilities available to GLaDOS right now.': 'capacités disponibles pour GLaDOS en ce moment.',
    'Tools run when called by a core. Inspect their inputs or run a fixed host command here. Local time and date are already supplied in context.':
      "Les outils s'exécutent à l'appel d'un noyau. Inspectez leurs entrées ou lancez ici une commande fixe. L'heure et la date locales sont déjà fournies dans le contexte.",
    'No callable tools reported. Connect to the engine to inspect capabilities.': 'Aucun outil signalé. Connectez le moteur pour inspecter les capacités.',
    'No description': 'Aucune description', 'Inspect tool': "Inspecter l'outil", 'Completed': 'Terminé', 'Failed': 'Échec',
    'No output': 'Aucune sortie', 'Command execution': 'Exécution de commande', 'Facility command queue unavailable.': 'File de commandes indisponible.',
    'Capacity unavailable': 'Capacité indisponible', 'Capacity unavailable.': 'Capacité indisponible.', 'Active': 'Actif', 'Idle': 'Inactif',
    'Reserved for interactive requests': 'Réservé aux demandes interactives', 'Shared': 'Partagé', 'Single step': 'Étape unique',
    'Preview only': 'Aperçu seulement', 'Selected action': 'Action choisie', 'Action': 'Action', 'Accepted': 'Accepté',
    'Full assistant with read-only tools': 'Assistant complet avec outils en lecture seule', 'Fallback used': 'Repli utilisé',
    'MCP server': 'Serveur MCP', 'Selected tools': 'Outils choisis', 'OPTION PREFERENCE': 'PRÉFÉRENCE DES OPTIONS',
    'In-flight requests': 'Demandes en cours', 'Available capacity': 'Capacité disponible', 'Queued requests': 'Demandes en file',
    'Interactive reservation': 'Réservation interactive', 'Shared capacity': 'Capacité partagée',
    'User interaction active · new background inference paused. Running requests may finish.':
      "Interaction en cours · nouvelles inférences de fond en pause. Les demandes en cours peuvent se terminer.",
    'No requests waiting.': 'Aucune demande en attente.', 'No update yet.': 'Aucune mise à jour pour le moment.',
    // Tasks
    'saved work and results.': 'travaux et résultats enregistrés.',
    'Create an assignment, ask GLaDOS to work on it, then inspect its result. Core state and reports live in Cores.':
      "Créez une mission, demandez à GLaDOS d'y travailler, puis consultez le résultat. L'état et les rapports des noyaux sont dans Noyaux.",
    'No summary': 'Aucun résumé', 'No report yet.': 'Aucun rapport pour le moment.', 'Reopen': 'Rouvrir', 'Mark done': 'Marquer terminé',
    'Sent to GLaDOS. The task updates when she saves a result.': 'Envoyé à GLaDOS. La tâche sera mise à jour quand elle enregistrera un résultat.',
    'Task updated.': 'Tâche mise à jour.', 'Task unavailable': 'Tâche indisponible',
    'Assignment saved and shared with GLaDOS.': 'Mission enregistrée et partagée avec GLaDOS.',
    // Context
    'Inspect GLaDOS’s inference context.': "Inspectez le contexte d'inférence de GLaDOS.", 'Core request': 'Requête du noyau',
    'Central Core · user request': 'Noyau central · demande utilisateur', 'Autonomy Core · autonomous check': "Noyau d'autonomie · vérification autonome",
    'Context source': 'Source du contexte', 'Live preview': 'Aperçu en direct', 'Last submitted request': 'Dernière requête envoyée',
    'Latest preview': 'Dernier aperçu', 'Latest request': 'Dernière requête', 'Refresh': 'Actualiser',
    'Live updates: ON': 'Mises à jour en direct : ON', 'Live updates: OFF': 'Mises à jour en direct : OFF', 'Download JSON': 'Télécharger le JSON',
    'Show all messages': 'Afficher tous les messages', 'Collapse all': 'Tout replier',
    'Messages are numbered in the order sent: reusable instructions, conversation history, live state, then the current input. Live preview shows what a new request would receive; new voice or text is added when it arrives. Last submitted request shows the actual payload. Tools are displayed separately below the messages.':
      "Les messages sont numérotés dans l'ordre d'envoi : consignes réutilisables, historique, état en direct, puis l'entrée actuelle. L'aperçu en direct montre ce que recevrait une nouvelle requête ; la voix ou le texte s'ajoute à l'arrivée. La dernière requête envoyée montre le contenu réel. Les outils sont affichés à part, sous les messages.",
    'Find in context': 'Chercher dans le contexte', 'Search prompts, core reports or conversation…': 'Chercher dans les consignes, rapports ou la conversation…',
    'Open the Neural Buffer to inspect the connected engine.': 'Ouvrez Contexte pour inspecter le moteur connecté.',
    'Available tool catalogue · preview only': 'Catalogue des outils · aperçu seulement', 'Tools offered in this request': 'Outils proposés dans cette requête',
    'This catalogue is not a message sent to GLaDOS. Routing selects the offered subset for the next request.':
      "Ce catalogue n'est pas un message envoyé à GLaDOS. Le routage choisit le sous-ensemble proposé à la prochaine requête.",
    'These definitions were sent in the separate tools field. They have no message number; their placement in the model template depends on the backend.':
      "Ces définitions ont été envoyées dans le champ « tools ». Elles n'ont pas de numéro ; leur place dans le modèle dépend du backend.",
    'Engine disconnected. Displayed context may be stale.': 'Moteur déconnecté. Le contexte affiché peut être périmé.',
    'Context could not be loaded.': "Le contexte n'a pas pu être chargé.", 'Loading context…': 'Chargement du contexte…',
    // Memory
    'Explore what GLaDOS remembers.': 'Explorez ce dont GLaDOS se souvient.', 'Recalled for this conversation': 'Rappelé pour cette conversation',
    'Waiting for a conversation topic.': "En attente d'un sujet de conversation.", 'Saved memory': 'Mémoire enregistrée',
    'Browse saved facts and conversation summaries. Searching here does not change what GLaDOS recalls.':
      'Parcourez les faits et résumés enregistrés. Chercher ici ne change pas ce que GLaDOS se rappelle.',
    'Search memory': 'Chercher dans la mémoire', 'A name, preference or project…': 'Un nom, une préférence ou un projet…',
    'Show': 'Afficher', 'Facts and summaries': 'Faits et résumés', 'Facts': 'Faits', 'Summaries': 'Résumés', 'Search': 'Chercher',
    'Open Test Chamber to browse saved memory.': 'Ouvrez Mémoire pour parcourir les souvenirs.', 'Previous': 'Précédent', 'Next': 'Suivant',
    'No saved memories match this view.': 'Aucun souvenir ne correspond à cette vue.', 'Summary': 'Résumé', 'Fact': 'Fait',
    'Date unavailable': 'Date indisponible', 'Engine disconnected. Displayed memory may be stale.': 'Moteur déconnecté. La mémoire affichée peut être périmée.',
    'Memory could not be loaded.': "La mémoire n'a pas pu être chargée.", 'Recall unavailable.': 'Rappel indisponible.',
    'The Memory Core is paused.': 'Le noyau Mémoire est en pause.', 'No saved facts match the current topic.': 'Aucun fait ne correspond au sujet actuel.',
    'Recall is disabled.': 'Le rappel est désactivé.', 'No saved memories in this view.': 'Aucun souvenir dans cette vue.',
    'Memory could not be read': "La mémoire n'a pas pu être lue", 'Memory change failed': 'La modification de la mémoire a échoué',
    // Settings
    'Interface appearance': "Apparence de l'interface",
    'Dark, Clinical, Terminal and PotatOS share the same live controls. This browser remembers your theme. Open the Presence screen from the top bar for the tablet view.':
      "Sombre, Clinique, Terminal et PotatOS partagent les mêmes commandes. Ce navigateur retient votre thème. Ouvrez l'Écran de présence depuis la barre du haut pour la vue tablette.",
    'Dark': 'Sombre', 'Clinical': 'Clinique', 'Terminal': 'Terminal', 'PotatOS': 'PotatOS',
    'Internet search sources': 'Sources de recherche Internet',
    'News requests read these sites’ pages directly, then search if needed. Other categories guide web searches. Enter one domain or site path per line, up to eight per category. Console changes apply to the next request and save to data/search_settings.yaml. File edits apply after restart.':
      "Les demandes d'actualité lisent directement ces sites, puis cherchent si besoin. Les autres catégories orientent les recherches web. Un domaine ou chemin par ligne, huit maximum par catégorie. Les changements faits ici s'appliquent à la prochaine requête et sont enregistrés dans data/search_settings.yaml. Les modifications du fichier s'appliquent après redémarrage.",
    'Weather': 'Météo', 'News': 'Actualités', 'Reddit communities': 'Communautés Reddit', 'General sources': 'Sources générales',
    'A favorite domain or site path': 'Un domaine ou chemin favori', 'Save favorite sources': 'Enregistrer les sources favorites',
    'Use at most eight sites per category.': 'Huit sites maximum par catégorie.',
    'Favorite search sources saved. They apply to the next search.': 'Sources favorites enregistrées. Elles s’appliquent à la prochaine recherche.',
    'This device as microphone': 'Cet appareil comme micro',
    'Talk to GLaDOS through this browser, for example a phone: its microphone replaces the PC microphone and GLaDOS speaks through it. The PC takes over again when you stop or close the page. While GLaDOS speaks here, this microphone is muted so she does not hear herself. Browsers allow the microphone only over HTTPS.':
      "Parlez à GLaDOS via ce navigateur, par exemple un téléphone : son micro remplace celui du PC et GLaDOS parle à travers lui. Le PC reprend la main quand vous arrêtez ou fermez la page. Pendant que GLaDOS parle ici, ce micro est coupé pour qu'elle ne s'entende pas. Les navigateurs n'autorisent le micro qu'en HTTPS.",
    'Stop using this device': "Ne plus utiliser cet appareil", 'Microphone in use: PC.': 'Micro utilisé : PC.',
    'Remote audio is not available with this audio backend.': "L'audio distant n'est pas disponible avec ce backend audio.",
    'Needs HTTPS: set webapp.tls: true and open the https:// address.': "HTTPS requis : activez webapp.tls: true et ouvrez l'adresse https://.",
    'Another device took over the microphone.': 'Un autre appareil a pris le micro.', 'Microphone permission was refused.': "L'accès au micro a été refusé.",
    'Remote audio failed.': "L'audio distant a échoué.",
    'Consultation lets GLaDOS ask external advisor AIs in the background. The base decides where Central Core runs: auto moves it to the API when this machine is too slow, local and api force it. Needs the API key named in the configuration. Changes apply immediately and save to data/nexus_settings.yaml.':
      "La consultation permet à GLaDOS d'interroger des IA conseillères externes en arrière-plan. La base décide où tourne le noyau central : auto le bascule vers l'API quand cette machine est trop lente, local et api l'imposent. Nécessite la clé API indiquée dans la configuration. Les changements s'appliquent tout de suite et sont enregistrés dans data/nexus_settings.yaml.",
    'Nexus Core unavailable.': 'Noyau Nexus indisponible.', 'Consultation': 'Consultation', 'On': 'Activé', 'Off': 'Désactivé',
    'Base model': 'Modèle de base', 'Auto': 'Auto', 'Local only': 'Local uniquement', 'API only': 'API uniquement',
    'Daily budget (USD)': 'Budget quotidien (USD)', 'Save Nexus settings': 'Enregistrer les réglages Nexus',
    'Choose a daily budget from 0 to 1000 USD.': 'Choisissez un budget quotidien entre 0 et 1000 USD.', 'Nexus settings saved.': 'Réglages Nexus enregistrés.',
    'autonomy, voice input and routing.': 'autonomie, entrée vocale et routage.', 'Create decision list': 'Créer une liste de décision',
    'Facility Sensors & Audio': 'Capteurs et audio du centre',
    'Choose the webcam, microphone and speakers connected to the GLaDOS host. Changes apply immediately for this session.':
      "Choisissez la webcam, le micro et les haut-parleurs branchés sur la machine de GLaDOS. Les changements s'appliquent tout de suite pour cette session.",
    'Webcam input': 'Webcam', 'Audio input': 'Entrée audio', 'Audio output': 'Sortie audio', 'Refresh devices': 'Actualiser les appareils',
    'Open Facility Settings to list connected devices.': 'Ouvrez Réglages pour lister les appareils.',
    'System default': 'Par défaut du système', 'Connect the engine to select host devices.': 'Connectez le moteur pour choisir les appareils.',
    'Listing connected devices…': 'Recherche des appareils…', 'Devices could not be updated.': "Les appareils n'ont pas pu être mis à jour.",
    'Connected devices refreshed.': 'Appareils actualisés.',
    'Motion guides each image analysis delay: a still scene favours the maximum, movement favours the minimum, with some randomness. Default: 2–5 seconds. Enter whole seconds; changes apply immediately and are saved across restarts. The blink follows actual image analysis.':
      "Le mouvement règle le délai entre deux analyses d'image : une scène immobile tend vers le maximum, le mouvement vers le minimum, avec un peu d'aléatoire. Par défaut : 2–5 secondes. Secondes entières ; appliqué tout de suite et conservé après redémarrage. Le clignement suit l'analyse réelle.",
    'Minimum delay (seconds)': 'Délai minimum (secondes)', 'Maximum delay (seconds)': 'Délai maximum (secondes)', 'Save interval range': "Enregistrer l'intervalle",
    'Choose whole seconds from 1 to 60, with minimum no greater than maximum.': 'Secondes entières de 1 à 60, minimum inférieur ou égal au maximum.',
    'Quiet mode': 'Mode silence',
    'Say “go to sleep” or “be quiet” to stop replies and suspend background cores. She keeps listening for “wake up”. Background sounds and unintelligible speech are ignored.':
      "Dites « go to sleep » ou « be quiet » pour arrêter les réponses et suspendre les noyaux de fond. Elle continue d'écouter « wake up ». Les bruits de fond et paroles inintelligibles sont ignorés.",
    'Reviews all core slots and recent conversation during idle pauses. Useful new alerts, completed searches and observed arrivals are passed to Central Core, which speaks in GLaDOS’s normal voice. Other cores run independently; silent checks back off automatically.':
      "Passe en revue les noyaux et la conversation récente pendant les pauses. Les alertes utiles, recherches terminées et arrivées observées sont transmises au noyau central, qui parle avec la voix habituelle de GLaDOS. Les autres noyaux tournent indépendamment ; les vérifications sans résultat s'espacent automatiquement.",
    'Capability routing': 'Routage des capacités',
    'GLaDOS manages routing and parallel replies automatically. Choose which list describes your available actions. Scores express preference among your options, not certainty.':
      "GLaDOS gère automatiquement le routage et les réponses parallèles. Choisissez la liste qui décrit vos actions disponibles. Les scores expriment une préférence entre options, pas une certitude.",
    'Voice input': 'Entrée vocale',
    'Optional transcripts use Gemma to keep a written version of your voice input. Routing, Emotion and replies reuse that text. Available in direct-audio mode; loads no Parakeet.':
      "Les transcriptions optionnelles utilisent Gemma pour garder une version écrite de votre voix. Le routage, l'émotion et les réponses réutilisent ce texte. Disponible en mode audio direct ; ne charge pas Parakeet.",
    'Transcripts': 'Transcriptions', 'Routing tree': 'Arbre de routage',
    'Choose the capability area, then its action. MCP services add a server choice before the tool. Click a decision to expand its choices; teal marks the latest route, yellow marks uncertainty.':
      "Choisissez le domaine, puis l'action. Les services MCP ajoutent le choix du serveur avant l'outil. Cliquez sur une décision pour déplier ses choix ; le turquoise marque la dernière route, le jaune l'incertitude.",
    'Saved. Applies to the next response.': "Enregistré. S'applique à la prochaine réponse.", 'Response instructions': 'Consignes de réponse',
    'Inspector updated.': 'Inspecteur mis à jour.', 'Camera schedule updated.': 'Planification de la caméra mise à jour.',
    'One observation queued.': 'Une observation en file.', 'Reconnect to the engine before making changes.': 'Reconnectez le moteur avant de modifier.',
    'The engine rejected the request.': 'Le moteur a refusé la demande.', 'Engine unavailable': 'Moteur indisponible',
    'Wake GLaDOS': 'Réveiller GLaDOS', 'Put GLaDOS to sleep': 'Endormir GLaDOS',
    'Quiet mode: listening for wake requests.': "Mode silence : en attente d'une demande de réveil.", 'GLaDOS is awake.': 'GLaDOS est réveillée.',
    'Autonomy enabled.': 'Autonomie activée.', 'Autonomy disabled.': 'Autonomie désactivée.', 'Enabled': 'Activé', 'Disabled': 'Désactivé',
    'Camera enabled.': 'Caméra activée.', 'Camera paused and released.': 'Caméra en pause et libérée.',
    'Active routing list saved.': 'Liste de routage active enregistrée.', 'Decision list deleted.': 'Liste de décision supprimée.',
    'Decision list saved.': 'Liste de décision enregistrée.', 'Enter valid JSON': 'Saisissez un JSON valide',
    'No route selected yet.': 'Aucune route choisie pour le moment.', 'No matching choice': 'Aucun choix correspondant',
    'Use audio of 30 seconds or less.': 'Utilisez un audio de 30 secondes maximum.', 'Audio file must be under 10 MB.': 'Le fichier audio doit faire moins de 10 Mo.',
    'Stop recording': "Arrêter l'enregistrement", 'Text mode': 'Mode texte', 'Preview complete. Nothing executed.': 'Aperçu terminé. Rien n’a été exécuté.',
    'Reply': 'Répondre', 'Ignore background speech': 'Ignorer les voix de fond', 'Ask for clarification': 'Demander une précision',
    'Sleep · listening for wake': 'Veille · écoute du réveil', 'Wake · resume replies': 'Réveil · reprise des réponses',
    'Assistant planning': "Planification de l'assistant", 'Configured fallback · no tool selected': 'Repli configuré · aucun outil choisi',
    'Assistant fills arguments': "L'assistant remplit les arguments",
    'Routing decision tree. Click a decision to expand or collapse its choices.': 'Arbre de décision du routage. Cliquez sur une décision pour déplier ou replier ses choix.',
    'Routing is automatic. Private drafts use spare capacity and are released only after routing and Emotion Core approve the reply.':
      "Le routage est automatique. Les brouillons privés utilisent la capacité restante et ne sont diffusés qu'après validation par le routage et le noyau Émotion.",
    'This model handles routing in its replies.': 'Ce modèle gère le routage dans ses réponses.',
    // Diagnostics
    'observability bus stream.': "flux du bus d'observabilité.", 'engine infrastructure, inference capacity, and device health.':
      "infrastructure du moteur, capacité d'inférence et état des appareils.",
    'Hardware & Model Adaptation': 'Matériel et adaptation du modèle', 'Reply latency': 'Latence de réponse',
    'Engine components': 'Composants du moteur', 'Audio': 'Audio', 'Interaction': 'Interaction', 'Vision': 'Vision',
    'MCP servers': 'Serveurs MCP', 'Inference diagnostics': "Diagnostics d'inférence", 'the routing, then conversation': 'le routage, puis la conversation',
    'background inference workers': "exécuteurs d'inférence de fond", 'Worker pool': "Groupe d'exécuteurs", 'Inspector': 'Inspecteur',
    'Last reply': 'Dernière réponse', 'Speech recognition': 'Reconnaissance vocale', 'Model first token': 'Premier jeton du modèle',
    'First voice clip': 'Premier extrait vocal', 'Until playback': "Jusqu'à la lecture", 'No reply measured yet': 'Aucune réponse mesurée',
    'Host Tier': 'Niveau de la machine', 'System RAM': 'RAM système', 'Adaptive Context': 'Contexte adaptatif', 'Max Output': 'Sortie max',
    'CPU Cores / Threads': 'Cœurs / threads CPU', 'Inference Timeout': "Délai d'inférence", 'Hardware telemetry standby': 'Télémétrie matérielle en attente',
    'No engine components reported.': 'Aucun composant signalé.', 'Microphone VAD': 'Détection de voix du micro',
    'Waiting for telemetry.': 'En attente de télémétrie.', 'Waiting for the engine to reconnect.': 'En attente de reconnexion du moteur.',
    'Listening only for a wake request.': "N'écoute qu'une demande de réveil.", 'Voice output active.': 'Voix active.',
    'Microphone voice activity detected.': 'Voix détectée au micro.',
    'No face detected. I\'ll wake when you return or speak.': 'Aucun visage détecté. Je me réveillerai quand vous reviendrez ou parlerez.',
    // Vision
    'Live webcam with face tracking': 'Webcam en direct avec suivi du visage', 'Waiting for the first observation.': 'En attente de la première observation.',
    'No previous observation yet.': 'Aucune observation précédente.', 'Waiting for timestamped observations.': "En attente d'observations horodatées.",
    'Paused schedule · observing once…': 'Planification en pause · observation unique…', 'Paused · camera released.': 'En pause · caméra libérée.',
    'Camera active · selecting fresh, sharp frames.': 'Caméra active · sélection d’images récentes et nettes.', 'Connecting to camera…': 'Connexion à la caméra…',
    'Resume Vision Core': 'Reprendre le noyau Vision', 'Suspend Vision Core': 'Suspendre le noyau Vision', 'No frame yet': 'Aucune image',
    'Ask GLaDOS a specific question to inspect the camera.': 'Posez une question précise à GLaDOS pour inspecter la caméra.',
    'No person detected': 'Personne détectée : aucune', 'Presence unclear': 'Présence incertaine',
    'Person present · face location unavailable': 'Personne présente · position du visage inconnue', 'Face in caption frame': 'Visage dans l’image analysée',
    'Face tracking': 'Suivi du visage', 'Compared with': 'Comparé à', 'First observation': 'Première observation',
    'Observation window': "Fenêtre d'observation", 'Caption rate': "Fréquence d'analyse", 'Webcam rate': 'Fréquence webcam',
    'Face detection': 'Détection du visage', 'Frame selection': "Sélection d'image", 'Focus score': 'Score de netteté',
    'Vision Core is unavailable in this profile.': "Le noyau Vision n'est pas disponible dans ce profil.", 'Camera: unavailable': 'Caméra : indisponible',
    'Suspend Vision Core observations and release the webcam.': 'Suspendre les observations et libérer la webcam.',
    'Resume Vision Core observations.': 'Reprendre les observations.', 'Waiting for an observation.': "En attente d'une observation.", 'Camera off.': 'Caméra éteinte.',
    'No extra response instructions. The base personality and tool instructions still apply.':
      "Aucune consigne supplémentaire. La personnalité de base et les consignes d'outils s'appliquent toujours.",
    'Response instructions are unavailable until the engine connects.': "Les consignes de réponse sont indisponibles tant que le moteur n'est pas connecté.",
    // Demo data
    'Watching the sky; weather briefing queued.': 'Surveille le ciel ; bulletin météo en file.', 'Hydration reminder armed. 45m cooldown.': "Rappel d'hydratation armé. 45 min de délai.",
    'Weather brief': 'Bulletin météo', 'Day agenda': 'Agenda du jour', 'Clash detected: sprint sync vs. dentist at 14:00.': 'Conflit : point sprint et dentiste à 14:00.',
    'Hydration nudge': "Rappel d'hydratation",
    'GLaDOS uses the PC microphone and speakers.': 'GLaDOS utilise le micro et les haut-parleurs du PC.',
    'This device is GLaDOS’s microphone and speaker. The PC microphone is ignored until you stop or close this page.':
      "Cet appareil sert de micro et de haut-parleur à GLaDOS. Le micro du PC est ignoré jusqu'à ce que vous arrêtiez ou fermiez cette page.",
    'Speaking': 'Parle', 'Listening': 'Écoute', 'Processing': 'Réfléchit', 'Sleeping': 'Endormie',
    'Paired devices': 'Appareils appairés',
    'Phones and other computers that opened the pairing link. Revoke one to sign it out; this PC never needs pairing. Run “glados pair” on the PC to show the link again.':
      "Téléphones et autres ordinateurs ayant ouvert le lien d'appairage. Révoquez-en un pour le déconnecter ; ce PC n'a jamais besoin d'appairage. Lancez « glados pair » sur le PC pour réafficher le lien.",
    'Sign out every device': 'Déconnecter tous les appareils', 'Revoke': 'Révoquer', 'this device': 'cet appareil',
    'No paired devices': 'Aucun appareil appairé', 'The console only listens on this machine.': "La console n'écoute que sur cette machine.",
    'Every device was signed out.': 'Tous les appareils ont été déconnectés.', 'Device signed out.': 'Appareil déconnecté.',
    'Working…': 'En cours…', 'Updated.': 'Mis à jour.', 'Download the GLaDOS certificate': 'Téléchargez le certificat GLaDOS',
    'and install it once on a phone to remove the security warning for good. Android: Settings › Security › Encryption & credentials › Install a certificate › CA certificate. iPhone: open the file, install the profile, then enable it under Settings › General › About › Certificate Trust Settings. It only covers local addresses, never public websites.':
      "et installez-le une fois sur le téléphone pour supprimer définitivement l'alerte de sécurité. Android : Paramètres › Sécurité › Chiffrement et identifiants › Installer un certificat › Certificat CA. iPhone : ouvrez le fichier, installez le profil, puis activez-le dans Réglages › Général › Informations › Réglages des certificats. Il ne couvre que les adresses locales, jamais les sites publics.",
    'capacity unknown': 'capacité inconnue', 'Reply latency trend': 'Évolution de la latence de réponse', 'End-of-speech silence': 'Silence de fin de phrase', 'Engine unavailable': 'Moteur indisponible',
  };
  // Strings that embed a live value.
  const FR_PATTERNS = [
    [/^(\d+) in-flight$/, '$1 en cours'], [/^in-flight (\d+)$/, 'en cours $1'], [/^shared processing (.*)$/, 'calcul partagé $1'],
    [/^UP (.*)$/, 'ACTIF $1'], [/^(MOTION|MOVEMENT) (.*)$/, 'MOUVEMENT $2'], [/^Median of last (\d+)$/, 'Médiane des $1 dernières'], [/^Quiet: (ON|OFF|connecting)$/, 'Silence : $1'],
    [/^Autonomy: (ON|OFF|connecting)$/, 'Autonomie : $1'], [/^Camera: (ON|OFF)$/, 'Caméra : $1'],
    [/^Reply started (\d+) ms after the user finished$/, 'Réponse commencée $1 ms après la fin de la phrase'],
    [/^Demo · (.+)$/, (_, state) => 'Démo · ' + (translate(state) ?? state)],
    [/^(\d+) inferences? in progress\.$/, '$1 inférence(s) en cours.'],
    [/^(\d+) \/ (\d+) channels active( · Background inference waiting for user response)?$/,
      (_, used, total, hold) => `${used} / ${total} canaux actifs` + (hold ? ' · Inférence de fond en attente de votre réponse' : '')],
    [/^Engine unavailable · (.*)$/, 'Moteur indisponible · $1'],
    [/^(.+) · before the timer starts$/, '$1 · avant le début du chrono'],
    [/^(.+) · GLaDOS \| Aperture Science$/, (_, view) => `${translate(view) ?? view} · GLaDOS | Aperture Science`],
    [/^(Microphone|Voice|Transcripts): (on|off)$/, (_, what, on) =>
      ({Microphone: 'Micro', Voice: 'Voix', Transcripts: 'Transcriptions'})[what] + ' : ' + (on === 'on' ? 'activé' : 'désactivé')],
  ];
  const LANGS = {en: 'English', fr: 'Français'};
  // Free text from GLaDOS, the user, memory and the event log stays as written.
  const SKIP = '#home-wire,#full-wire,#context-sections,#memory-facts,#memory-recall,pre,code,textarea,script,style,[data-no-i18n]';
  const ATTRS = ['placeholder', 'title', 'aria-label'];
  const texts = new WeakMap();  // text node -> {original, shown}
  let lang = 'en';

  const translate = text => {
    if (lang !== 'fr') return null;
    const key = text.replace(/\s+/g, ' ').trim();
    if (!key) return null;
    if (Object.hasOwn(FR, key)) return FR[key];
    for (const [pattern, replacement] of FR_PATTERNS) if (pattern.test(key)) return key.replace(pattern, replacement);
    return null;
  };
  const skipped = node => {
    const element = node.nodeType === Node.ELEMENT_NODE ? node : node.parentElement;
    return !element || !!element.closest(SKIP);
  };
  const textNode = node => {
    if (skipped(node)) return;
    const seen = texts.get(node);
    if (seen && node.data === seen.shown) return;  // our own write
    const original = node.data, translated = translate(original);
    if (translated === null) { texts.delete(node); return; }
    const shown = original.replace(original.trim(), () => translated);
    texts.set(node, {original, shown});
    node.data = shown;
  };
  const attributes = element => {
    if (skipped(element)) return;
    for (const name of ATTRS) {
      if (!element.hasAttribute(name)) continue;
      const stash = 'data-i18n-' + name, value = element.getAttribute(name);
      if (element.getAttribute(stash + '-shown') === value) continue;
      const translated = translate(value);
      if (translated === null) { element.removeAttribute(stash); continue; }
      element.setAttribute(stash, value);
      element.setAttribute(stash + '-shown', translated);
      element.setAttribute(name, translated);
    }
  };
  const walk = root => {
    if (root.nodeType === Node.TEXT_NODE) return textNode(root);
    if (root.nodeType !== Node.ELEMENT_NODE) return;
    attributes(root);
    const tree = document.createTreeWalker(root, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
    for (let node = tree.nextNode(); node; node = tree.nextNode()) node.nodeType === Node.TEXT_NODE ? textNode(node) : attributes(node);
  };
  const restore = () => {
    const tree = document.createTreeWalker(document.documentElement, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
    for (let node = tree.nextNode(); node; node = tree.nextNode()) {
      if (node.nodeType === Node.TEXT_NODE) {
        const seen = texts.get(node);
        if (seen && node.data === seen.shown) node.data = seen.original;
        texts.delete(node);
      } else for (const name of ATTRS) {
        const stash = 'data-i18n-' + name;
        if (node.hasAttribute(stash) && node.getAttribute(name) === node.getAttribute(stash + '-shown')) node.setAttribute(name, node.getAttribute(stash));
        node.removeAttribute(stash); node.removeAttribute(stash + '-shown');
      }
    }
  };
  const apply = next => {
    lang = LANGS[next] ? next : 'en';
    document.documentElement.lang = lang;
    try { localStorage.setItem('glados-lang', lang); } catch { /* private mode */ }
    lang === 'fr' ? (walk(document.body), document.querySelector('title') && walk(document.querySelector('title'))) : restore();
  };
  const picker = () => {
    const themes = document.querySelector('.theme-label');
    if (!themes || document.getElementById('lang-choice')) return;
    const label = document.createElement('label');
    label.className = 'theme-label';
    const select = document.createElement('select');
    select.id = 'lang-choice';
    select.setAttribute('aria-label', 'Language');
    for (const [code, name] of Object.entries(LANGS)) select.add(new Option(name, code, false, code === lang));
    select.onchange = () => apply(select.value);
    label.append(document.createTextNode('Language '), select);
    themes.after(label);
  };

  let saved = null;
  try { saved = localStorage.getItem('glados-lang'); } catch { /* private mode */ }
  lang = saved || ((navigator.language || '').slice(0, 2) === 'fr' ? 'fr' : 'en');
  window.GladosI18n = {apply, translate: text => translate(text) ?? text, get lang() { return lang; }, dictionary: {fr: FR}};
  const start = () => {
    picker();
    apply(lang);
    const observer = new MutationObserver(records => {
      if (lang !== 'fr') return;
      for (const record of records) {
        if (record.type === 'characterData') textNode(record.target);
        else if (record.type === 'attributes') attributes(record.target);
        else record.addedNodes.forEach(walk);
      }
    });
    observer.observe(document.body, {subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ATTRS});
    // The tab title follows the current view.
    const title = document.querySelector('title');
    if (title) { walk(title); observer.observe(title, {subtree: true, childList: true, characterData: true}); }
  };
  document.readyState === 'loading' ? document.addEventListener('DOMContentLoaded', start) : start();
})();
