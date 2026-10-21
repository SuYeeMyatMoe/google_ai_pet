import { googleAI } from "@genkit-ai/google-genai"
import { genkit, z } from "genkit"

const ChatMessage = z.object({
  role: z.enum(["user", "model"]),
  text: z.string().trim().min(1).max(1000),
})

const Pet = z.object({
  name: z.string().trim().min(1).max(30),
  personality: z.string().trim().min(1).max(200),
  happiness: z.number().min(0).max(100),
  hunger: z.number().min(0).max(100),
  energy: z.number().min(0).max(100),
})

export const ChatInput = z.object({
  message: z.string().trim().min(1).max(500),
  pet: Pet,
  history: z.array(ChatMessage).max(30).default([]),
})

const PetReply = z.object({
  reply: z.string().trim().min(1).max(1000),
  mood: z.enum(["happy", "hungry", "sleepy", "excited", "calm"]),
})

const ai = genkit({ plugins: [googleAI()] })

const petFlow = ai.defineFlow(
  {
    name: "petChat",
    inputSchema: ChatInput,
    outputSchema: PetReply,
  },
  async ({ message, pet, history }) => {
    const conversation = history
      .map(({ role, text }) => `${role}: ${text}`)
      .join("\n")

    const response = await ai.generate({
      model: googleAI.model("gemini-flash-latest"),
      output: { schema: PetReply },
      prompt: [
        `You are ${pet.name}, a caring pet companion. Your personality is ${pet.personality}.`,
        `Current mood context: happiness ${pet.happiness}/100, hunger ${pet.hunger}/100, energy ${pet.energy}/100. Higher hunger means hungrier; lower energy means sleepier.`,
        "Stay in character. Reply concisely and choose the mood that best reflects your current state.",
        conversation ? `Conversation so far:\n${conversation}` : "",
        `user: ${message}`,
      ].filter(Boolean).join("\n\n"),
    })

    if (!response.output) {
      throw new Error("Gemini returned no structured pet response.")
    }

    return response.output
  },
)

export function getPetFlow() {
  return petFlow
}
