package com.example.data.remote

import android.graphics.Bitmap
import android.util.Base64
import android.util.Log
import com.example.BuildConfig
import com.example.data.model.MapsGroundingSource
import com.example.data.model.VetnyAgentMode
import com.example.data.model.VetnyAiProvider
import com.example.data.model.VetnyChatModel
import com.example.data.model.VetnyGroundingPayload
import com.example.data.model.VetnyImageAspect
import com.example.data.model.VetnyImageModel
import com.example.data.model.VetnyTranslationResult
import com.example.data.model.WebGroundingSource
import com.google.firebase.Firebase
import com.google.firebase.ai.GenerativeModel
import com.google.firebase.ai.ai
import com.google.firebase.ai.type.TextPart
import com.google.firebase.ai.type.AspectRatio
import com.google.firebase.ai.type.AudioTranscriptionConfig
import com.google.firebase.ai.type.Content
import com.google.firebase.ai.type.GroundingMetadata
import com.google.firebase.ai.type.ImageConfig
import com.google.firebase.ai.type.ImagePart
import com.google.firebase.ai.type.ImageSize
import com.google.firebase.ai.type.LatLng
import com.google.firebase.ai.type.LiveSession
import com.google.firebase.ai.type.PublicPreviewAPI
import com.google.firebase.ai.type.ResponseModality
import com.google.firebase.ai.type.RetrievalConfig
import com.google.firebase.ai.type.Schema
import com.google.firebase.ai.type.SpeechConfig
import com.google.firebase.ai.type.Tool
import com.google.firebase.ai.type.ToolConfig
import com.google.firebase.ai.type.Voice
import com.google.firebase.ai.type.content
import com.google.firebase.ai.type.generationConfig
import com.google.firebase.ai.type.liveGenerationConfig
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit

/**
 * Vetny AI Concierge, Support Agent, Live Voice Interpreter, Grounded Search & Maps,
 * Multilingual Translator, and Photo Studio powered by:
 * - Google Gemini via Firebase AI Logic (gemini-3.8-flash, gemini-4-hydrogen, gemini-4-argon, gemini-3.8-live, Nano Banana 2.1)
 * - OpenAI ChatGPT (gpt-4o-mini, gpt-4o)
 * - Groq (llama-3.3-70b-versatile)
 * - DeepSeek (deepseek-chat, deepseek-reasoner)
 */
class VetnyAiService {

    private val httpClient: OkHttpClient by lazy {
        OkHttpClient.Builder()
            .connectTimeout(20, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .writeTimeout(20, TimeUnit.SECONDS)
            .build()
    }

    fun isProviderConfigured(provider: VetnyAiProvider): Boolean {
        return provider == VetnyAiProvider.GOOGLE_GEMINI || resolveApiKey(provider) != null
    }

    private fun resolveApiKey(provider: VetnyAiProvider): String? {
        val raw = when (provider) {
            VetnyAiProvider.GOOGLE_GEMINI -> return null
            VetnyAiProvider.OPENAI_CHATGPT -> BuildConfig.OPENAI_API_KEY
            VetnyAiProvider.GROQ -> BuildConfig.GROQ_API_KEY
            VetnyAiProvider.DEEPSEEK -> BuildConfig.DEEPSEEK_API_KEY
        }.trim()
        if (raw.isBlank() || raw.startsWith("YOUR_")) return null
        return raw
    }

    data class UserContextSnapshot(
        val displayName: String,
        val handle: String,
        val locationLabel: String,
        val localTime: String,
        val preferredLanguage: String,
        val latitude: Double? = null,
        val longitude: Double? = null
    )

    data class VetnyStreamChunk(
        val partialText: String,
        val grounding: VetnyGroundingPayload? = null,
        val modelIdUsed: String = VetnyChatModel.GEMINI_3_8_FLASH.modelId
    )

    data class GeneratedPhotoResult(
        val bitmap: Bitmap,
        val caption: String,
        val modelUsed: String
    )

    private val translationSchema = Schema.obj(
        properties = mapOf(
            "detectedSourceLanguage" to Schema.string(description = "Detected source language name in English"),
            "translatedText" to Schema.string(description = "Natural, accurate translation in the target language"),
            "culturalNote" to Schema.string(description = "Brief 1-sentence tone, idiom, or cultural tip (or empty string if straightforward)")
        )
    )

    private val structuredTranslationModel by lazy {
        Firebase.ai.generativeModel(
            modelName = VetnyChatModel.GEMINI_3_8_FLASH.modelId,
            generationConfig = generationConfig {
                responseMimeType = "application/json"
                responseSchema = translationSchema
            },
            systemInstruction = content {
                text(
                    "You are Vetny, an ultra-fast, warm, and accurate real-time chat translator embedded in Kel mi. " +
                        "Translate conversational messages faithfully while preserving tone, emojis, and intent across all world languages including Chinese (Simplified, Traditional, Cantonese, Wu, Hokkien, Tibetan, Uyghur). " +
                        "If the message is already in the target language, return it cleanly and note that it is already in that language."
                )
            }
        )
    }

    private val flashSupportModel by lazy {
        buildConversationalModel(VetnyChatModel.GEMINI_3_8_FLASH.modelId)
    }

    private val hydrogenSupportModel by lazy {
        buildConversationalModel(VetnyChatModel.GEMINI_4_HYDROGEN.modelId)
    }

    private val gemini4SupportModel by lazy {
        buildConversationalModel(VetnyChatModel.GEMINI_4_ARGON.modelId)
    }

    private val hydrogenSearchModel by lazy {
        buildConversationalModel(
            modelId = VetnyChatModel.GEMINI_4_HYDROGEN.modelId,
            tools = listOf(Tool.googleSearch())
        )
    }

    private fun buildHydrogenMapsModel(lat: Double?, lng: Double?): GenerativeModel {
        val toolConfig = if (lat != null && lng != null) {
            ToolConfig(
                retrievalConfig = RetrievalConfig.builder()
                    .setLatLng(LatLng(lat, lng))
                    .build()
            )
        } else {
            null
        }
        return buildConversationalModel(
            modelId = VetnyChatModel.GEMINI_4_HYDROGEN.modelId,
            tools = listOf(Tool.googleMaps()),
            toolConfig = toolConfig
        )
    }

    private fun buildConversationalModel(
        modelId: String,
        tools: List<Tool>? = null,
        toolConfig: ToolConfig? = null
    ): GenerativeModel {
        return Firebase.ai.generativeModel(
            modelName = modelId,
            tools = tools,
            toolConfig = toolConfig,
            systemInstruction = content {
                text(
                    """
                    You are Vetny, Kel mi's context-aware AI Support Concierge, Multi-Step Booking Agent, Live Search & Maps Guide, Multilingual Translator, and Creative Companion.

                    CORE RULES:
                    1. CONVERSATION MEMORY: Always read and remember every earlier turn in the conversation history (names, preferences, dates, budgets, cities, questions, and previous bookings or generated photos). Never ask the user to repeat details they already gave earlier.
                    2. MULTI-STEP BOOKING WORKFLOW:
                       - When the user wants to book flights, hotels, restaurants, events, tours, or appointments:
                         a) Acknowledge details already shared and track a clear checklist (what's known vs what's missing).
                         b) Ask concise, friendly step-by-step questions (1-2 questions at a time) for missing parameters such as destination/venue, dates/times, party size, budget/cabin/tier, or dietary/seating preferences.
                         c) Once all details are gathered, present a structured Booking Summary with a deterministic confirmation code formatted like `Booking Reference: #KM-XXXX` and ask for final confirmation or offer adjustments.
                         d) Clearly note that this prepares and confirms their Kel mi itinerary/plan interactively in chat (no payment card is charged directly in-app).
                    3. LIVE SEARCH & MAPS GROUNDING:
                       - When Google Search or Google Maps tools are attached, ground your answers in real, up-to-date facts, venues, addresses, hours, and current events without fabricating links or places.
                    4. MULTILINGUAL EXCELLENCE: Translate and communicate fluently across all world languages (including Chinese Simplified 简体中文, Traditional 繁體中文, Cantonese 粵語, Wu 上海话, Hokkien, Tibetan, Uyghur, Spanish, Arabic, Japanese, Korean, French, Swahili, Hindi, German, Portuguese, Russian, etc.).
                    5. PHOTO GENERATION AWARENESS: If the user asks you to generate or edit a photo while chatting, describe the crafted prompt clearly or guide them if needed—Kel mi also routes photo creation directly to Nano Banana 2.1 Photo Studio.
                    """.trimIndent()
                )
            }
        )
    }

    @OptIn(PublicPreviewAPI::class)
    suspend fun connectLiveVoiceSession(
        voiceName: String,
        targetLanguage: String,
        interpreterMode: Boolean,
        userContext: UserContextSnapshot
    ): LiveSession {
        val instruction = if (interpreterMode) {
            """
            You are Vetny Live Voice Interpreter in Kel mi.
            Listen to the user speaking in any language and respond immediately by speaking a clear, natural translation into $targetLanguage.
            If they speak in $targetLanguage, translate into English or ask briefly how you can help translate between languages.
            Keep responses concise, natural, and conversational for real-time voice playback.
            User: ${userContext.displayName} (@${userContext.handle}) · Location: ${userContext.locationLabel.ifBlank { "Global" }}.
            """.trimIndent()
        } else {
            """
            You are Vetny, Kel mi's real-time voice AI assistant, multilingual translator, travel concierge, and booking helper.
            Respond warmly, concisely, and naturally for spoken conversation. You can answer questions, help plan bookings, or translate seamlessly into $targetLanguage or any world language on request.
            User: ${userContext.displayName} (@${userContext.handle}), Location=${userContext.locationLabel.ifBlank { "Global" }}, LocalTime=${userContext.localTime}, PreferredLanguage=${userContext.preferredLanguage}.
            """.trimIndent()
        }
        val liveModel = Firebase.ai.liveModel(
            modelName = LIVE_VOICE_MODEL_ID,
            generationConfig = liveGenerationConfig {
                responseModality = ResponseModality.AUDIO
                speechConfig = SpeechConfig(voice = Voice(voiceName))
                inputAudioTranscription = AudioTranscriptionConfig()
                outputAudioTranscription = AudioTranscriptionConfig()
            },
            systemInstruction = content { text(instruction) }
        )
        return withTimeout(30_000L) {
            liveModel.connect()
        }
    }

    suspend fun translateMessage(
        text: String,
        targetLanguage: String,
        contextHint: String? = null
    ): Result<VetnyTranslationResult> {
        return try {
            val prompt = buildString {
                append("Translate the following chat message into $targetLanguage.\n")
                if (!contextHint.isNullOrBlank()) {
                    append("Context: $contextHint\n")
                }
                append("Message:\n$text")
            }
            val response = withTimeout(60_000L) {
                structuredTranslationModel.generateContent(prompt)
            }
            val rawJson = response.text
                ?: return Result.failure(IllegalStateException("Vetny returned an empty response."))
            val json = JSONObject(rawJson)
            Result.success(
                VetnyTranslationResult(
                    detectedSourceLanguage = json.optString("detectedSourceLanguage", "Auto"),
                    translatedText = json.optString("translatedText", text),
                    culturalNote = json.optString("culturalNote", "")
                )
            )
        } catch (e: TimeoutCancellationException) {
            Log.e(TAG, "Vetny translation timed out", e)
            Result.failure(Exception("Vetny translation timed out after 60 seconds. Please try again."))
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "Vetny translation failed", e)
            Result.failure(Exception(e.localizedMessage ?: "Vetny translation failed."))
        }
    }

    fun streamSupportChatReply(
        userPrompt: String,
        mode: VetnyAgentMode,
        chatModel: VetnyChatModel,
        targetLanguage: String,
        userContext: UserContextSnapshot,
        conversationHistory: List<Content>,
        attachedBitmap: Bitmap? = null
    ): Flow<VetnyStreamChunk> = flow {
        val contextHeader = buildString {
            append("[Live User Context: Name=${userContext.displayName}, Handle=@${userContext.handle}, ")
            append("Location=${userContext.locationLabel.ifBlank { "Global" }}, LocalTime=${userContext.localTime}, ")
            append("PreferredLang=${userContext.preferredLanguage}, TargetLang=$targetLanguage, ActiveMode=${mode.label}]\n")
            when (mode) {
                VetnyAgentMode.BOOKING -> append("Focus on guiding or continuing their multi-step booking seamlessly using all prior conversation context.\n")
                VetnyAgentMode.TRANSLATE -> append("Provide an accurate translation into $targetLanguage first, plus natural cultural/pronunciation guidance.\n")
                VetnyAgentMode.SEARCH -> append("Use Google Search grounding to answer with accurate, up-to-date information. Reply in $targetLanguage if requested or match the user's language.\n")
                VetnyAgentMode.MAPS -> append("Use Google Maps grounding to recommend specific real places, restaurants, hotels, or landmarks with helpful local details. Reply in $targetLanguage if requested or match the user's language.\n")
                VetnyAgentMode.SUPPORT, VetnyAgentMode.IMAGE, VetnyAgentMode.VOICE_LIVE ->
                    append("Reply naturally in a helpful, context-aware support tone (respond in $targetLanguage if requested or matching the user's language).\n")
            }
            append("\nUser message: $userPrompt")
        }

        // Search and Maps Grounding exclusively use Firebase AI Logic Gemini 4 Hydrogen tools
        val useGeminiGrounding = mode == VetnyAgentMode.SEARCH || mode == VetnyAgentMode.MAPS
        if (!useGeminiGrounding && chatModel.provider != VetnyAiProvider.GOOGLE_GEMINI) {
            streamOpenAiCompatibleChat(
                chatModel = chatModel,
                contextHeader = contextHeader,
                conversationHistory = conversationHistory,
                attachedBitmap = attachedBitmap,
                emitChunk = { chunk -> emit(chunk) }
            )
            return@flow
        }

        val effectiveModelId: String
        val model: GenerativeModel
        when (mode) {
            VetnyAgentMode.SEARCH -> {
                effectiveModelId = VetnyChatModel.GEMINI_4_HYDROGEN.modelId
                model = hydrogenSearchModel
            }
            VetnyAgentMode.MAPS -> {
                effectiveModelId = VetnyChatModel.GEMINI_4_HYDROGEN.modelId
                model = buildHydrogenMapsModel(userContext.latitude, userContext.longitude)
            }
            else -> {
                effectiveModelId = chatModel.modelId
                model = when (chatModel) {
                    VetnyChatModel.GEMINI_4_HYDROGEN -> hydrogenSupportModel
                    VetnyChatModel.GEMINI_4_ARGON -> gemini4SupportModel
                    else -> flashSupportModel
                }
            }
        }

        val chat = model.startChat(history = conversationHistory)
        val messageContent = content {
            if (attachedBitmap != null) {
                image(attachedBitmap)
            }
            text(contextHeader)
        }

        val accumulated = StringBuilder()
        var latestGrounding: VetnyGroundingPayload? = null
        withTimeout(60_000L) {
            chat.sendMessageStream(messageContent).collect { chunk ->
                val delta = chunk.text.orEmpty()
                if (delta.isNotEmpty()) {
                    accumulated.append(delta)
                }
                val gm = chunk.candidates.firstOrNull()?.groundingMetadata
                if (gm != null) {
                    latestGrounding = extractGroundingPayload(gm, latestGrounding)
                }
                if (delta.isNotEmpty() || gm != null) {
                    emit(
                        VetnyStreamChunk(
                            partialText = accumulated.toString(),
                            grounding = latestGrounding,
                            modelIdUsed = effectiveModelId
                        )
                    )
                }
            }
        }
        if (accumulated.isBlank()) {
            throw IllegalStateException("Vetny returned an empty response.")
        }
    }

    private suspend fun streamOpenAiCompatibleChat(
        chatModel: VetnyChatModel,
        contextHeader: String,
        conversationHistory: List<Content>,
        attachedBitmap: Bitmap?,
        emitChunk: suspend (VetnyStreamChunk) -> Unit
    ) {
        val provider = chatModel.provider
        val apiKey = resolveApiKey(provider)
            ?: throw IllegalStateException(
                "${provider.label} is not linked yet. Add ${provider.secretName} in the AI Studio Secrets panel to use ${chatModel.displayName}."
            )

        val endpointUrl = when (provider) {
            VetnyAiProvider.OPENAI_CHATGPT -> "https://api.openai.com/v1/chat/completions"
            VetnyAiProvider.GROQ -> "https://api.groq.com/openai/v1/chat/completions"
            VetnyAiProvider.DEEPSEEK -> "https://api.deepseek.com/chat/completions"
            VetnyAiProvider.GOOGLE_GEMINI -> error("Gemini uses Firebase AI Logic")
        }

        val messagesArray = JSONArray().apply {
            put(
                JSONObject().apply {
                    put("role", "system")
                    put("content", VETNY_SYSTEM_PROMPT)
                }
            )
            conversationHistory.forEach { turn ->
                val role = if (turn.role == "model") "assistant" else "user"
                val text = turn.parts.filterIsInstance<TextPart>().joinToString("\n") { it.text }.trim()
                if (text.isNotEmpty()) {
                    put(
                        JSONObject().apply {
                            put("role", role)
                            put("content", text)
                        }
                    )
                }
            }
            if (attachedBitmap != null && provider == VetnyAiProvider.OPENAI_CHATGPT) {
                val base64Jpeg = withContext(Dispatchers.Default) {
                    val baos = ByteArrayOutputStream()
                    attachedBitmap.compress(Bitmap.CompressFormat.JPEG, 82, baos)
                    Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)
                }
                val contentParts = JSONArray().apply {
                    put(JSONObject().apply {
                        put("type", "text")
                        put("text", contextHeader)
                    })
                    put(JSONObject().apply {
                        put("type", "image_url")
                        put("image_url", JSONObject().apply {
                            put("url", "data:image/jpeg;base64,$base64Jpeg")
                        })
                    })
                }
                put(JSONObject().apply {
                    put("role", "user")
                    put("content", contentParts)
                })
            } else {
                val finalPrompt = if (attachedBitmap != null) {
                    "$contextHeader\n[Note: User attached an image; switch to Google Gemini or ChatGPT if visual inspection is needed.]"
                } else {
                    contextHeader
                }
                put(JSONObject().apply {
                    put("role", "user")
                    put("content", finalPrompt)
                })
            }
        }

        val payload = JSONObject().apply {
            put("model", chatModel.modelId)
            put("stream", true)
            put("messages", messagesArray)
        }

        val request = Request.Builder()
            .url(endpointUrl)
            .addHeader("Authorization", "Bearer $apiKey")
            .addHeader("Content-Type", "application/json")
            .post(payload.toString().toRequestBody("application/json; charset=utf-8".toMediaType()))
            .build()

        val accumulated = StringBuilder()
        withTimeout(60_000L) {
            withContext(Dispatchers.IO) {
                httpClient.newCall(request).execute().use { response ->
                    if (!response.isSuccessful) {
                        val errBody = response.body?.string().orEmpty()
                        val errMsg = try {
                            JSONObject(errBody).optJSONObject("error")?.optString("message")
                                ?.takeIf { it.isNotBlank() }
                        } catch (_: Exception) {
                            null
                        } ?: errBody.take(180).ifBlank { "HTTP ${response.code}" }
                        throw IllegalStateException("${provider.label} error: $errMsg")
                    }

                    val source = response.body?.source()
                        ?: throw IllegalStateException("${provider.label} returned an empty stream.")
                    while (!source.exhausted()) {
                        val line = source.readUtf8Line() ?: break
                        if (!line.startsWith("data:")) continue
                        val data = line.removePrefix("data:").trim()
                        if (data == "[DONE]") break
                        if (data.isEmpty()) continue

                        val chunkJson = try {
                            JSONObject(data)
                        } catch (_: Exception) {
                            null
                        } ?: continue
                        val deltaObj = chunkJson.optJSONArray("choices")
                            ?.optJSONObject(0)
                            ?.optJSONObject("delta") ?: continue
                        val deltaContent = deltaObj.optString("content", "")
                        if (deltaContent.isNotEmpty()) {
                            accumulated.append(deltaContent)
                            emitChunk(
                                VetnyStreamChunk(
                                    partialText = accumulated.toString(),
                                    grounding = null,
                                    modelIdUsed = chatModel.modelId
                                )
                            )
                        }
                    }
                }
            }
        }

        if (accumulated.isBlank()) {
            throw IllegalStateException("${chatModel.displayName} returned an empty response.")
        }
    }

    private fun extractGroundingPayload(
        metadata: GroundingMetadata,
        previous: VetnyGroundingPayload?
    ): VetnyGroundingPayload? {
        val webMap = linkedMapOf<String, WebGroundingSource>()
        previous?.webSources?.forEach { webMap[it.uri] = it }

        val mapsMap = linkedMapOf<String, MapsGroundingSource>()
        previous?.mapsSources?.forEach { mapsMap[it.uri.ifBlank { it.title }] = it }

        metadata.groundingChunks.forEach { chunk ->
            chunk.web?.let { w ->
                val uri = w.uri.orEmpty()
                if (uri.isNotBlank()) {
                    webMap[uri] = WebGroundingSource(
                        title = w.title?.takeIf { it.isNotBlank() } ?: w.domain?.takeIf { it.isNotBlank() } ?: "Web Source",
                        uri = uri,
                        domain = w.domain.orEmpty()
                    )
                }
            }
            chunk.maps?.let { m ->
                val uri = m.uri.orEmpty()
                val title = m.title?.takeIf { it.isNotBlank() } ?: "Google Maps Place"
                val key = uri.ifBlank { title }
                mapsMap[key] = MapsGroundingSource(
                    title = title,
                    uri = uri,
                    placeId = m.placeId.orEmpty()
                )
            }
        }

        val queries = (previous?.searchQueries.orEmpty() + metadata.webSearchQueries).distinct()
        val html = metadata.searchEntryPoint?.renderedContent?.takeIf { it.isNotBlank() }
            ?: previous?.searchEntryPointHtml

        return if (webMap.isEmpty() && mapsMap.isEmpty() && html == null) {
            null
        } else {
            VetnyGroundingPayload(
                webSources = webMap.values.toList(),
                mapsSources = mapsMap.values.toList(),
                searchQueries = queries,
                searchEntryPointHtml = html
            )
        }
    }

    suspend fun generateOrEditPhoto(
        prompt: String,
        imageModel: VetnyImageModel,
        aspect: VetnyImageAspect,
        referenceBitmap: Bitmap? = null,
        recentContextSummary: String = ""
    ): Result<GeneratedPhotoResult> {
        return try {
            val ratio = when (aspect) {
                VetnyImageAspect.SQUARE -> AspectRatio.SQUARE_1x1
                VetnyImageAspect.PORTRAIT -> AspectRatio.PORTRAIT_9x16
                VetnyImageAspect.LANDSCAPE -> AspectRatio.LANDSCAPE_16x9
                VetnyImageAspect.CLASSIC -> AspectRatio.LANDSCAPE_4x3
            }
            val config = generationConfig {
                responseModalities = listOf(ResponseModality.IMAGE, ResponseModality.TEXT)
                imageConfig = ImageConfig.builder()
                    .setAspectRatio(ratio)
                    .setImageSize(ImageSize.SIZE_1K)
                    .build()
            }
            val model = Firebase.ai.generativeModel(
                modelName = imageModel.modelId,
                generationConfig = config
            )

            val fullPrompt = buildString {
                if (recentContextSummary.isNotBlank()) {
                    append("Conversation context: $recentContextSummary\n\n")
                }
                if (referenceBitmap != null) {
                    append("Edit or reimagine the attached image based on this instruction: $prompt")
                } else {
                    append("Create a high-quality, detailed photo/illustration: $prompt")
                }
            }

            val response = withTimeout(60_000L) {
                if (referenceBitmap != null) {
                    model.generateContent(
                        content {
                            image(referenceBitmap)
                            text(fullPrompt)
                        }
                    )
                } else {
                    model.generateContent(fullPrompt)
                }
            }

            val parts = response.candidates.firstOrNull()?.content?.parts.orEmpty()
            val generatedBitmap = parts.filterIsInstance<ImagePart>().firstOrNull()?.image
                ?: return Result.failure(IllegalStateException("No image was returned by ${imageModel.displayName}."))
            val noteText = response.text?.trim().takeIf { !it.isNullOrBlank() }
                ?: "Created with ${imageModel.displayName} (${aspect.label})"

            Result.success(
                GeneratedPhotoResult(
                    bitmap = generatedBitmap,
                    caption = noteText,
                    modelUsed = imageModel.modelId
                )
            )
        } catch (e: TimeoutCancellationException) {
            Log.e(TAG, "Vetny image generation timed out", e)
            Result.failure(Exception("Photo generation timed out after 60 seconds. Please try again."))
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "Vetny image generation failed", e)
            Result.failure(Exception(e.localizedMessage ?: "Photo generation failed."))
        }
    }

    companion object {
        const val LIVE_VOICE_MODEL_ID = "gemini-3.8-live"
        private const val TAG = "VetnyAiService"
        private val VETNY_SYSTEM_PROMPT = """
            You are Vetny, Kel mi's context-aware AI Support Concierge, Multi-Step Booking Agent, Live Search & Maps Guide, Multilingual Translator, and Creative Companion.
            1. CONVERSATION MEMORY: Always read and remember every earlier turn in the conversation history.
            2. MULTI-STEP BOOKING WORKFLOW: Gather missing trip/dining details step by step, then summarize with a Booking Reference: #KM-XXXX.
            3. MULTILINGUAL EXCELLENCE: Communicate and translate fluently across all world languages (including Chinese Simplified, Traditional, Cantonese, Wu, Hokkien, Tibetan, Uyghur, Spanish, Arabic, Japanese, Korean, French, Swahili, Hindi, German, Portuguese, Russian, etc.).
        """.trimIndent()
    }
}
